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
import shell_tensor


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
        fxp_fractional_bits=1,
        weight_dtype=tf.int64,
    ):
        self.units = int(units)
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.skip_normalization = skip_normalization
        self.fxp_fractional_bits = fxp_fractional_bits
        self.weight_dtype = weight_dtype

        self.built = False
        self.weights = []

    def build(self, input_shape):
        self.units_in = int(input_shape[1])
        self.kernel = self.kernel_initializer([self.units_in, self.units])

        # Convert kernel to fixed point.
        self.kernel_initializer = tf.cast(self.kernel, self.weight_dtype)
        self.weights.append(self.kernel)

        if self.use_bias:
            self.bias = self.bias_initializer([self.units])
            self.bias = tf.cast(self.bias, self.weight_dtype)
            self.weights.append(self.bias)
        else:
            self.bias = None

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

    def backward(self, dy, rotation_key, is_first_layer=False):
        """dense backward"""
        x = self._layer_input
        z = self._layer_intermediate
        y = self._layer_output
        kernel = self.weights[0]
        grad_weights = []
        batch_size = int(x.shape[0])

        if self.activation_deriv is not None:
            dy = self.activation_deriv(z, dy)

        if isinstance(dy, shell_tensor.ShellTensor64):
            # It is a good idea to reduce the multiplication count before
            # the multiplication with the kernel. Multiplying by the kernel
            # requires a reduce_sum operation which makes it easy to exceed
            # the plaintext modulus by the fixed point fractional bits, if
            # the multiplication count is too high.
            dy = dy.get_at_multiplication_count(0)

        if is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            kernel_t = tf.transpose(kernel)
            d_x = shell_tensor.matmul(dy, kernel_t)

        # Perform the fixed point multiplication.
        d_weights = shell_tensor.matmul(tf.transpose(x), dy, rotation_key)

        if not self.skip_normalization:
            assert False, "Normalization not implemented yet."
            # d_weights = d_weights / batch_size
        grad_weights.append(d_weights)

        if self.use_bias:
            assert False, "Bias not implemented yet"
            # TODO(jchoncholas): reduce_sum is very expensive and requires slot rotation.
            # Not implemented yet. A better way than the reduce sum is to set batch size to 1 less
            # and use that last slot as the bias with input 1.
            # #d_bias = shell_tensor.reduce_sum(dy, axis=0)
            # if not self.skip_normalization:
            #     d_bias = d_bias / batch_size
            # grad_weights.append(d_bias)

        return grad_weights, d_x
