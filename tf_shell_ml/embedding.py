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
        skip_embeddings_below_index=0,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.skip_embeddings_below_index = skip_embeddings_below_index

        self.weights = []
        self.build()

    def build(self):
        self.embeddings = tf.Variable(
            self.embeddings_initializer([self.input_dim, self.output_dim])
        )
        self.weights.append(self.embeddings)
        self.built = True

    def __call__(self, inputs):
        if inputs.dtype != tf.int64:
            raise ValueError(
                f"Embedding layer expects int64 input. Got {inputs.dtype}."
            )
        if inputs.ndim != 2:
            raise ValueError(f"Embedding layer expects rank 2 input. Got {inputs}.")

        self._layer_input = inputs
        outputs = tf.experimental.numpy.take(self.embeddings, inputs, axis=0)
        return outputs

    def backward(self, dy, rotation_key):
        """
        dy is shape (batch_size, sentence_length, output_dimension)
        _layer_input is (batch_size, sentence_length)

        This function sums all the dy[a, b, :] where _layer_input[a, b] has the
        same value.

        There are two ways to do the ciphertext aggregation:
          1) rotate -> add -> mask
              First rotate the value selected by the index to slot 0,
              then add the result to a running tally for this index. Once
              everything has been added, mask off everything except the first
              slot, and decrypt.
          2) mask -> add -> reduce sum
              Mask off the value selected by the index with a one-hot vector.
              Add the result to a running tally for this index. Lastly,
              reduce_sum over the batch dimension and decrypt.
        Since rotations are the slowest of the 3 operations, I think it would
        be faster to do option 2, since values in the same slot can be added
        requiring only one rotation.

        The op we need for this layer is a special version of tf.segment_sum()
        and some extra logic to properly handle the batching dimenmsion.
        tf-shell uses batch axis packing, meaning dy is really shape
        (input_dimension, output_dimension) and each element is a ciphertext
        with (2*batch_size) slots. tf_shell.segment_sum must pull apart the
        packing dimension of the values by masking with a one-hot.
        """
        batch_size = self._layer_input.shape[0] // 2

        if dy.ndim != self._layer_input.ndim + 1:
            raise ValueError(
                f"Embedding layer dy ndims exptected {self._layer_input.ndim + 1}. Got {dy}."
            )
        sentence_len = dy.shape[1]

        indices = self._layer_input
        values = dy

        indices = tf.where(
            indices < self.skip_embeddings_below_index,
            tf.constant(-1, dtype=indices.dtype),
            indices,
        )

        summedvalues, self._last_slot_count = tf_shell.segment_sum(
            values,
            indices,
            self.input_dim,
            rotation_key,
        )

        return [summedvalues], tf.zeros(0)

    def unpack(self, plaintext_packed_dx):
        batch_size = tf.shape(plaintext_packed_dx)[0] // 2
        return plaintext_packed_dx[0, 0] + plaintext_packed_dx[batch_size, 1]
