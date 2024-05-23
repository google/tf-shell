#!/usr/bin/python
#
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from tensorflow.python.keras import initializers
import tf_shell


class ShellEmbedding:
    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        self.weights = []
        self.grad_map = None
        self.build()

    def build(self):
        self.embeddings = tf.Variable(
            self.embeddings_initializer([self.input_dim, self.output_dim])
        )
        self.weights.append(self.embeddings)
        self.built = True

    def _reset_grad_map(self):
        # The gradient map stores temerary gradients between microbatches.
        #
        # Ideally the MutableHashTable would store values of type ShellTensor64
        # but it does not support ExtensionTypes. Instead, store the raw variant
        # tensor held inside the ShellTensor64 class.
        #
        # Even worse, the MutableHashTable does not support variant tensors
        # which are non-scalar (more than one dimension).
        # see: https://github.com/tensorflow/tensorflow/blob/7b077cd4a83bcf3e296796bc7a32ff9e68a10fa2/tensorflow/core/kernels/lookup_table_op.cc#L1188
        # So instead, use a MutableDenseHashTable.

        default_row = tf_shell.to_shell_plaintext(
            tf.zeros([self._shell_context.num_slots, self.output_dim]),
            self._shell_context,
        )._raw_tensor

        self.grad_map = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.variant,
            default_value=default_row,
            empty_key=-1,
            deleted_key=-2,
        )
        self.grad_map_count = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.int64, value_dtype=tf.int64, default_value=0
        )

    def __call__(self, inputs):
        if inputs.dtype != tf.int64:
            raise ValueError(
                f"Embedding layer expects int64 input. Got {inputs.dtype}."
            )
        if inputs.ndim != 2:
            raise ValueError(
                f"Embedding layer expects rank 2 input. Got {inputs}."
            )

        self._layer_input = inputs
        outputs = tf.experimental.numpy.take(self.embeddings, inputs, axis=0)
        return outputs

    def _add_or_insert_grad_row(self, row_index, shell_tensor):
        """Add a gradient row to the hash map if it does not exist, otherwise
        add the gradient to the existing row.
        """
        c = self.grad_map_count.lookup(row_index)

        if c == tf.constant(0, dtype=tf.int64):
            self.grad_map_count.insert(row_index, 1)
            self.grad_map.insert(row_index, shell_tensor._raw_tensor)
        else:
            self.grad_map_count.insert(row_index, c + 1)

            raw_row = self.grad_map.lookup(row_index)
            # raw_row shape: (batch_size, output_dim)

            # First create a ShellTensor from the variant raw_tensor
            # stored in the hash map.
            shell_row = tf_shell.ShellTensor64(
                _raw_tensor=raw_row,
                _context=self._shell_context,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=self._scaling_factor,
                _is_enc=self._is_enc,
                _noise_bit_count=self._orig_noise_bit_count + tf.cast(c, tf.int32),
            )
            row_sum = shell_row + shell_tensor
            self.grad_map.insert(row_index, row_sum._raw_tensor)

    def backward_accum(self, dy, rotation_key):
        """Accumulate the gradients for the embedding layer. Unlike the
        other layers backward() methods, this does not return gradients.

        The tricky party about backpropagation of encrypted gradients through
        the embedding layer is that each sample in dy must be applied to a
        row of the embedding matrix. This requires splitting the batch-axis
        packing of dy.

        The strategy is to copy dy `batch_size` times, and rotate each so b'th
        copy has sample b in packing position zero. Next, the samples are
        grouped by which row of the embedding matrix they are applied to
        (which is based on self._layer_input).
        """

        self._shell_context = dy._context
        self._underlying_dtype = dy._underlying_dtype
        self._scaling_factor = dy._scaling_factor
        self._is_enc = dy._is_enc
        self._orig_noise_bit_count = dy._noise_bit_count
        # dy = tf_shell.to_shell_plaintext(dy, self.shell_context)

        if self.grad_map is None:
            self._reset_grad_map()

        # tfshell uses batch axis packing (two batches per ciphertext).
        batch_size = self._layer_input.shape[0] // 2

        # if len(dy.shape) == tf.constant(2):
        #     skip_word_dim = True
        #     sentence_len = 1
        # else:
        #     skip_word_dim = False
        #     sentence_len = dy.shape[1]

        if dy.ndim != self._layer_input.ndim + 1:
            raise ValueError(
                f"Embedding layer dy ndims exptected {self._layer_input.ndim + 1}. Got {dy}."
            )
        sentence_len = dy.shape[1]

        for word in tf.range(sentence_len):
            # For every sample in the batch, rotate dy so that the sample's grad is
            # in the first position, and sum based on layer_input. Store the running
            # total in grad_map.
            for b in tf.range(batch_size):
                # if skip_word_dim:
                #    dy_b = tf_shell.roll(dy, -b, rotation_key)
                # else:
                dy_b = tf_shell.roll(dy[:, word], -b, rotation_key)

                # Examine the row of `_layer input` to determine which row of
                # the embedding matrix to apply the gradient to. Note this is a
                # scalar of type integer and there is one for each batch (and
                # two batches per ciphertext).
                # if skip_word_dim:
                #    row_index_bottom = self._layer_input[b]
                #    row_index_top = self._layer_input[b + batch_size]
                # else:
                row_index_bottom = self._layer_input[b, word]
                row_index_top = self._layer_input[b + batch_size, word]

                self._add_or_insert_grad_row(row_index_bottom, dy_b)
                self._add_or_insert_grad_row(row_index_top, dy_b)

    def _lookup_shell_tensor(self, row_index, count, secret_key):
        grad_row = self.grad_map.lookup(row_index)

        # Turn the grad variant tensor back into a ShellTensor64.
        grad_row = tf_shell.ShellTensor64(
            _raw_tensor=grad_row,
            _context=self._shell_context,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self._is_enc,
            _noise_bit_count=self._orig_noise_bit_count + tf.cast(count, tf.int32),
        )

        # Decrypt and expand dims to [1, output_dim]
        grad_row = tf_shell.to_tensorflow(grad_row, secret_key)
        grad_row = tf.expand_dims(grad_row[0, :], axis=0)
        return grad_row

    def decrypt_grad(self, secret_key):
        """Get the accumulated gradients in matrix form which can then be
        applied to the weights. This method returns two gradients.
        """

        if self.grad_map is None:
            return None

        # Start building the gradient tensor.
        c = self.grad_map_count.lookup(0)
        if c == tf.constant(0, dtype=tf.int64):
            # Initialize if just begining and no grad update for the
            # first row exists.
            grads = tf.zeros([1, self.output_dim], dtype=self.embeddings.dtype)
        else:
            grads = self._lookup_shell_tensor(0, c, secret_key)

        # Build up the gradient tensor row by row where each row corresponds to
        # a sample in the batch.
        for i in tf.range(1, self.input_dim, dtype=tf.int64):
            tf.autograph.experimental.set_loop_options(
                maximum_iterations=self.input_dim,
                shape_invariants=[(grads, tf.TensorShape([None, self.output_dim]))]
            )

            c = self.grad_map_count.lookup(i)

            # If no gradient was accumulated for this row, add a row of zeros.
            if c == tf.constant(0, dtype=tf.int64):
                grads = tf.concat([grads, tf.zeros([1, self.output_dim])], axis=0)
            else:
                grad_row = self._lookup_shell_tensor(i, c, secret_key)
                grads = tf.concat([grads, grad_row], axis=0)

        # Reset the gradient accumulator and return.
        self.grad_map = None
        return grads
