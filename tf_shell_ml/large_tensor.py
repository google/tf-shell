import tensorflow as tf
import tf_shell

# TensorFlow uses GRPC to communicate between nodes. The maximum message size
# in GRPC is 4GB (UINT32_MAX). In order to avoid exceeding this limit, we need
# to split large tensors into smaller chunks. This file provides functions to
# do so.
#
# As written, the maximum number of splits is limited by MAX_NUM_SPLITS due to
# not knowing how large ciphertexts are until graph execution time (due to
# autocontext).

UINT32_MAX = 4294967295  # Maximum size for GRPC
SAFETY_FACTOR = 0.9  # Leave some headroom below the limit
MAX_NUM_SPLITS = 10  # Maximum number of splits


def calculate_tf_shell_split_sizes(context, total_elements):
    """
    Calculate split sizes of a ShellTensor that don't exceed GRPC limit.

    Args:
        total_elements: Total number of elements in tensor
        dtype: Data type of tensor

    Returns:
        List of split sizes that sum to total_elements
    """

    # Each element in the shell tensor is a tuple of polynomials, one for
    # each component of the ciphertext, which have `ring degree` elements.
    # In tf_shell, these are represented with uint64_t values. Serialziation
    # also includes the power_of_s (int) and the error (double).
    bytes_per_element = (
        tf.cast(context.num_slots, dtype=tf.int64)
        * 2
        * (tf.size(context.main_moduli, out_type=tf.int64) * 8 + 4 + 8)
    )
    max_elements = tf.cast(
        tf.constant(int(UINT32_MAX * SAFETY_FACTOR), dtype=tf.int64)
        / bytes_per_element,
        dtype=tf.int64,
    )

    num_full_splits = total_elements // max_elements
    remainder = total_elements % max_elements

    split_sizes = tf.fill([num_full_splits], max_elements)

    split_sizes = tf.cond(
        remainder > 0,
        lambda: tf.concat([split_sizes, [remainder]], axis=0),
        lambda: tf.identity(split_sizes),
    )

    zeros = tf.cond(
        remainder > 0,
        lambda: tf.fill(
            [MAX_NUM_SPLITS - num_full_splits - 1], tf.constant(0, dtype=tf.int64)
        ),
        lambda: tf.fill(
            [MAX_NUM_SPLITS - num_full_splits], tf.constant(0, dtype=tf.int64)
        ),
    )

    split_sizes = tf.concat([split_sizes, zeros], axis=0)
    return split_sizes


def calculate_split_sizes(total_elements, dtype):
    """
    Calculate split sizes or a TensorFlow tensor that don't exceed GRPC limit.

    Args:
        total_elements: Total number of elements in tensor
        dtype: Data type of tensor

    Returns:
        List of split sizes that sum to total_elements
    """
    bytes_per_element = tf.cast(dtype.size, dtype=tf.int64)

    max_elements = tf.cast(
        tf.constant(int(UINT32_MAX * SAFETY_FACTOR), dtype=tf.int64)
        / bytes_per_element,
        dtype=tf.int64,
    )

    num_full_splits = total_elements // max_elements
    remainder = total_elements % max_elements

    # Create list of split sizes
    split_sizes = tf.fill([num_full_splits], max_elements)

    split_sizes = tf.cond(
        remainder > 0,
        lambda: tf.concat([split_sizes, [remainder]], axis=0),
        lambda: tf.identity(split_sizes),
    )

    zeros = tf.cond(
        remainder > 0,
        lambda: tf.fill(
            [MAX_NUM_SPLITS - num_full_splits - 1], tf.constant(0, dtype=tf.int64)
        ),
        lambda: tf.fill(
            [MAX_NUM_SPLITS - num_full_splits], tf.constant(0, dtype=tf.int64)
        ),
    )

    split_sizes = tf.concat([split_sizes, zeros], axis=0)
    return split_sizes


def split_tensor(tensor):
    """
    Split a large tensor into smaller chunks using tf.split with uneven splits.

    Args:
        tensor: Input tensor to be split

    Returns:
        Tuple containing:
        - List of tensor chunks
        - Metadata dictionary with original shape and other info needed for reassembly
    """
    if isinstance(tensor, tf_shell.ShellTensor64):
        shape = tf_shell.shape(tensor)
        total_elements = tensor._raw_tensor.shape.num_elements()

        # Calculate split sizes
        split_sizes = calculate_tf_shell_split_sizes(tensor._context, total_elements)

        # Reshape tensor to 1D for splitting, ignoring the batch dimension
        flat_tensor = tf_shell.reshape(tensor, [tensor._context.num_slots, -1])

        # Split into chunks of calculated sizes
        chunks = tf_shell.split(flat_tensor, split_sizes, axis=1, num=MAX_NUM_SPLITS)

    else:
        shape = tf.shape(tensor)
        total_elements = tf.reduce_prod(tf.cast(shape, tf.int64))

        # Calculate split sizes
        split_sizes = calculate_split_sizes(total_elements, tensor.dtype)

        # Reshape tensor to 1D for splitting.
        flat_tensor = tf.reshape(tensor, [-1])

        # Split into chunks of calculated sizes
        chunks = tf_shell.split(flat_tensor, split_sizes, axis=0, num=MAX_NUM_SPLITS)

    metadata = {
        "original_shape": shape,
        "split_sizes": split_sizes,
    }

    return chunks, metadata


def split_tensor_list(tensors):
    """
    Split a list of tensors into chunks.

    Args:
        tensors: List of input tensors

    Returns:
        Tuple containing:
        - List of lists of tensor chunks (one list per input tensor)
        - List of metadata dictionaries (one per input tensor)
    """
    all_chunks = []
    all_metadata = []

    for tensor in tensors:
        chunks, metadata = split_tensor(tensor)
        all_chunks.append(chunks)
        all_metadata.append(metadata)

    return all_chunks, all_metadata


def reassemble_tensor(chunks, metadata):
    """
    Reassemble a tensor from its chunks and metadata.

    Args:
        chunks: List of tensor chunks
        metadata: Dictionary containing original shape and other assembly info

    Returns:
        Reassembled tensor with original shape
    """
    # Concatenate chunks
    if isinstance(chunks[0], tf_shell.ShellTensor64):
        flat_tensor = tf_shell.concat(chunks, axis=1)
    else:
        flat_tensor = tf.concat(chunks, axis=0)

    # Reshape to original shape
    original_shape = metadata["original_shape"]
    reassembled_tensor = tf_shell.reshape(flat_tensor, original_shape)

    return reassembled_tensor


def reassemble_tensor_list(all_chunks, all_metadata):
    """
    Reassemble a list of tensors from their chunks and metadata.

    Args:
        all_chunks: List of lists of tensor chunks
        all_metadata: List of metadata dictionaries

    Returns:
        List of reassembled tensors
    """
    reassembled_tensors = []

    for chunks, metadata in zip(all_chunks, all_metadata):
        reassembled_tensor = reassemble_tensor(chunks, metadata)
        reassembled_tensors.append(reassembled_tensor)

    return reassembled_tensors
