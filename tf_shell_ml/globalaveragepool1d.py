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


class GlobalAveragePooling1D(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, training=False):
        if training:
            self._layer_intermediate = inputs.shape[1]

        outputs = tf_shell.reduce_sum(inputs, axis=1)
        outputs /= inputs.shape[1]

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
                "activation_deriv": self.activation_deriv,
            }
        )
        return config

    def backward(self, dy):
        dx = tf_shell.expand_dims(dy, axis=1)
        dx = tf_shell.broadcast_to(
            dx, (dx.shape[0], self._layer_intermediate, dx.shape[2])
        )
        return [], dx

    def unpack(self, packed_dw):
        return packed_dw
