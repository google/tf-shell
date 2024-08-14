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


class ShellDropout:
    def __init__(
        self,
        rate,
        noise_shape=None,
        seed=None,
        per_batch=False,
    ):
        self.rate = float(rate)
        if (self.rate < 0.0) or (self.rate >= 1.0):
            raise ValueError(
                "rate must be a float in the range [0, 1), got {}".format(self.rate)
            )
        self.noise_shape = noise_shape
        self.seed = seed
        self.per_batch = per_batch
        if self.per_batch and self.noise_shape is not None:
            raise ValueError("noise_shape must be None when per_batch is True")

        self.built = False

    def build(self, input_shape):
        self.units_in = int(input_shape[1])

        self.built = True

    def __call__(self, inputs, training=False):
        if not self.built:
            self.build(inputs.shape)

        if not training or self.rate == 0.0:
            return inputs

        if self.per_batch:
            dummy_input = tf.ones([1] + inputs.shape[1:])
        else:
            dummy_input = tf.ones(inputs.shape)

        dropout_mask = tf.nn.dropout(
            dummy_input,
            self.rate,
            noise_shape=self.noise_shape,
            seed=self.seed,
        )

        self._layer_intermediate = dropout_mask
        self.outputs = inputs * dropout_mask
        return self.outputs

    def backward(self, dy):
        d_x = dy * self._layer_intermediate
        return [], d_x
