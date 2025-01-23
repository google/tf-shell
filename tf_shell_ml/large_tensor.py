import tensorflow as tf
import tf_shell

# TensorFlow uses GRPC to communicate between nodes. The maximum message size
# in GRPC is 4GB (UINT32_MAX). In order to avoid exceeding this limit, we need
# to split large tensors into smaller chunks. This file provides functions to
# do so.
#
# As written, the maximum number of splits is limited by MAX_NUM_SPLITS due to
# not knowing how large ciphertexts are until graph execution time (due to
# autocontext). TensorFlow must know the number of splits at graph construction
# time. While this has the downside of imposing an upper bound on the largest
# tensors that can be sent between nodes, the alternative (dynamically
# determining the number of splits at runtime) requires padding the last tensor
# with zeros, which wastes bandwidth (up to 4GB, per tensor).

UINT32_MAX = 4294967295  # Maximum size for GRPC
SAFETY_FACTOR = 0.9 / 4  # Leave some headroom below the limit
# Warning: When the message size is exceeded, TensorFlow will segfault with no
# stack trace or other debugging info. While the code below correctly computes
# an upper bound of the size of a ciphertext, TensorFlow adds overhead that
# cannot be accounted for (or at least, I don't know how to account for it). As
# such the SAFETY_FACTOR is set very low, which unfortunately limits the size of
# ciphertexts that can be sent between nodes.
MAX_NUM_SPLITS = 100  # Maximum number of splits


def calculate_tf_shell_split_sizes(context, total_elements):
    """
    Calculate split sizes of a ShellTensor that don't exceed GRPC limit.

    Args:
        total_elements: Total number of elements in tensor
        dtype: Data type of tensor

    Returns:
        List of split sizes that sum to total_elements
    """
    num_slots = tf.cast(context.num_slots, dtype=tf.int64)
    num_main_moduli = tf.size(context.main_moduli, out_type=tf.int64)

    # Each element in the shell tensor is a tuple of polynomials, one for
    # each component of the ciphertext, which have `ring degree` elements.
    # In tf_shell, these are represented with uint64_t values. Serialziation
    # also includes the power_of_s (int) and the error (double).
    # The real serialization code in SHELL takes into account the size of the
    # moduli, where the code below is an upper bound (assuming the moduli are
    # all 64-bits).
    sizeof_uint64 = 8
    num_components = 2  # TODO: increases by 1 for every ct*ct multiplication.
    bytes_per_component = (
        4  # log_n (int)
        + 1  # is_ntt (bool)
        + num_main_moduli * num_slots * sizeof_uint64  # coeff vectors
    )
    extra_bytes = 4 + 8  # power_of_s (int) and error (double)
    bytes_per_ct = num_components * bytes_per_component + extra_bytes
    tf.print("PYTHON bytes_per_ct: ", bytes_per_ct)

    max_elements_per_tensor = tf.cast(
        tf.constant(int(UINT32_MAX * SAFETY_FACTOR), dtype=tf.int64) / bytes_per_ct,
        dtype=tf.int64,
    )

    num_full_splits = total_elements // max_elements_per_tensor
    remainder = total_elements % max_elements_per_tensor

    # Create list of splits which have valid elements.
    split_sizes = tf.fill([num_full_splits], max_elements_per_tensor)
    split_sizes = tf.cond(
        remainder > 0,
        lambda: tf.concat([split_sizes, [remainder]], axis=0),
        lambda: tf.identity(split_sizes),
    )

    # Pad the empty splits with zeros.
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
    with tf.name_scope("large_tensor_split"):
        if isinstance(tensor, tf_shell.ShellTensor64):
            shape = tf_shell.shape(tensor)
            total_elements = tensor._raw_tensor.shape.num_elements()

            # Calculate split sizes
            split_sizes = calculate_tf_shell_split_sizes(
                tensor._context, total_elements
            )

            # Reshape tensor to 1D for splitting, ignoring the batch dimension
            flat_tensor = tf_shell.reshape(tensor, [tensor._context.num_slots, -1])

            # Split into chunks of calculated sizes
            chunks = tf_shell.split(
                flat_tensor, split_sizes, axis=1, num=MAX_NUM_SPLITS
            )

        else:
            shape = tf.shape(tensor)
            total_elements = tf.reduce_prod(tf.cast(shape, tf.int64))

            # Calculate split sizes
            split_sizes = calculate_split_sizes(total_elements, tensor.dtype)

            # Reshape tensor to 1D for splitting.
            flat_tensor = tf.reshape(tensor, [-1])

            # Split into chunks of calculated sizes
            chunks = tf_shell.split(
                flat_tensor, split_sizes, axis=0, num=MAX_NUM_SPLITS
            )

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
    with tf.name_scope("large_tensor_reassemble"):
        # Concatenate chunks
        if isinstance(chunks[0], tf_shell.ShellTensor64):
            flat_tensor = tf_shell.concat(chunks, axis=1)
        else:
            flat_tensor = tf.concat(chunks, axis=0)

        # Reshape to original shape
        reassembled_tensor = tf_shell.reshape(flat_tensor, metadata["original_shape"])

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
