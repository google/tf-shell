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
import tensorflow.keras as keras
from tensorflow.python.keras import initializers
import tf_shell
from math import prod


class Flatten(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
                "activation_deriv": self.activation_deriv,
            }
        )
        return config

    def build(self, input_shape):
        self.flat_shape = [prod(input_shape[1:])]

    def call(self, inputs, training=False):
        self.batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, [self.batch_size] + self.flat_shape)

    def backward(self, dy, rotation_key=None):
        # On the forward pass, inputs may be batched differently than the
        # ciphertext scheme when not in eager mode. Pad them to match the
        # ciphertext scheme.
        if isinstance(dy, tf_shell.ShellTensor64):
            new_shape = list(self.input_shape)
            new_shape[0] = dy._context.num_slots
        else:
            new_shape = self.input_shape
        dw = []
        dx = tf_shell.reshape(dy, new_shape)
        return dw, dx
