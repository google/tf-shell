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


class ShellDense:
    def __init__(
        self,
        units,
        activation=None,
        activation_deriv=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        skip_normalization=True,
        weight_dtype=tf.float32,
        is_first_layer=False,
    ):
        self.units = int(units)
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.skip_normalization = skip_normalization
        self.weight_dtype = weight_dtype
        self.is_first_layer = is_first_layer

        self.built = False
        self.weights = []

    def build(self, input_shape):
        self.units_in = int(input_shape[1])
        self.kernel = tf.Variable(self.kernel_initializer([self.units_in, self.units]))
        self.weights.append(self.kernel)

        if self.use_bias:
            self.bias = tf.Variable(self.bias_initializer([self.units]))
            self.weights.append(self.bias)

        self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        self._layer_input = inputs

        if self.use_bias:
            outputs = tf.matmul(inputs, self.weights[0]) + self.weights[1]
        else:
            outputs = tf.matmul(inputs, self.weights[0])

        self._layer_intermediate = outputs

        if self.activation is not None:
            outputs = self.activation(outputs)

        self._layer_output = outputs

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
        d_weights = tf_shell.matmul(tf.transpose(x), dy, rotation_key)

        if not self.skip_normalization:
            d_weights = d_weights / batch_size
        grad_weights.append(d_weights)

        if self.use_bias:
            d_bias = tf_shell.reduce_sum(dy, axis=0, rotation_key=rotation_key)
            if not self.skip_normalization:
                d_bias = d_bias / batch_size
            grad_weights.append(d_bias)

        return grad_weights, d_x
