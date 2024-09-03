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


# For details see https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07
class Conv2D:
    def __init__(
        self,
        units,
        in_channels,
        out_channels,
        strides=1,
        activation=None,
        activation_deriv=None,
        kernel_initializer="glorot_uniform",
        weight_dtype=tf.float32,
        is_first_layer=False,
        use_fast_reduce_sum=False,
    ):
        self.units = int(units)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.strides = [1, int(strides), int(strides), 1]
        self.activation = activation
        self.activation_deriv = activation_deriv

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.weight_dtype = weight_dtype
        self.is_first_layer = is_first_layer
        self.use_fast_reduce_sum = use_fast_reduce_sum

        self.built = False
        self.weights = []

    def build(self, input_shape):
        self.units_in = int(input_shape[1])
        self.kernel = tf.Variable(
            self.kernel_initializer(
                [self.units, self.units, self.in_channels, self.out_channels]
            )
        )
        self.weights.append(self.kernel)
        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)

        self._layer_input = inputs

        outputs = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding="SAME")

        self._layer_intermediate = outputs

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def backward(self, dy, rotation_key):
        """dense backward"""
        x = self._layer_input
        z = self._layer_intermediate
        y = self._layer_output
        kernel = self.weights[0]
        grad_weights = []
        batch_size = int(x.shape[0])

        if self.activation_deriv is not None:
            dy = self.activation_deriv(z, dy)

        if self.is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            # Perform the multiplication for dy/dx.
            kernel_t = tf.transpose(kernel)
            d_x = tf_shell.matmul(dy, kernel_t)

        # Perform the multiplication for dy/dw.
        if self.use_fast_reduce_sum:
            d_weights = tf_shell.matmul(tf.transpose(x), dy, fast=True)
        else:
            d_weights = tf_shell.matmul(tf.transpose(x), dy, rotation_key)

        grad_weights.append(d_weights)

        return grad_weights, d_x

    def unpack(self, plaintext_packed_dx):
        batch_size = plaintext_packed_dx.shape[0] // 2
        return plaintext_packed_dx[0] + plaintext_packed_dx[batch_size]
