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
from tf_shell_ml.activation import (
    serialize_activation,
    deserialize_activation,
)


class ShellDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        activation_deriv=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        is_first_layer=False,
        grad_reduction="none",
    ):
        super().__init__()
        self.units = int(units)
        self.activation = deserialize_activation(activation)
        self.activation_deriv = deserialize_activation(activation_deriv)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.is_first_layer = is_first_layer
        self.grad_reduction = grad_reduction

        if grad_reduction not in ["galois", "fast", "none"]:
            raise ValueError(
                f"Invalid grad_reduction type: {grad_reduction} (must be 'galois', 'fast', or 'none')"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": serialize_activation(self.activation),
                "activation_deriv": serialize_activation(self.activation_deriv),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "is_first_layer": self.is_first_layer,
                "grad_reduction": self.grad_reduction,
            }
        )
        return config

    def build(self, input_shape):
        self.units_in = int(input_shape[1])
        self.kernel = self.add_weight(
            shape=[self.units_in, self.units],
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=tf.keras.backend.floatx(),
            name="kernel",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.units],
                initializer="zeros",
                trainable=True,
                dtype=tf.keras.backend.floatx(),
                name="kernel",
            )

    def call(self, inputs, training=False):
        if training:
            self._layer_input = inputs

        if self.use_bias:
            outputs = tf.matmul(inputs, self.weights[0]) + self.weights[1]
        else:
            outputs = tf.matmul(inputs, self.weights[0])

        self._layer_intermediate = outputs

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def backward(self, dy, rotation_key=None):
        """dense backward"""
        x = self._layer_input
        z = self._layer_intermediate
        kernel = self.weights[0]
        d_ws = []

        # On the forward pass, inputs may be batched differently than the
        # ciphertext scheme when not in eager mode. Pad them to match the
        # ciphertext scheme.
        if isinstance(dy, tf_shell.ShellTensor64):
            padding = [[0, dy._context.num_slots - x.shape[0]]] + [
                [0, 0] for _ in range(len(x.shape) - 1)
            ]
            x = tf.pad(x, padding)

        if self.activation_deriv is not None:
            dy = self.activation_deriv(z, dy)

        if self.is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            # Perform the multiplication for dy/dx.
            kernel_t = tf.transpose(kernel)
            d_x = tf_shell.matmul(dy, kernel_t)

        # Perform the multiplication for dy/dw.
        d_w = tf_shell.matmul(
            tf.transpose(x),
            dy,
            rotation_key,
            pt_ct_reduction=self.grad_reduction,
            emulate_pt_ct=True,
        )
        d_ws.append(d_w)

        if self.use_bias:
            if self.grad_reduction == "galois":
                d_bias = tf_shell.reduce_sum(dy, axis=0, rotation_key=rotation_key)
            elif self.grad_reduction == "fast":
                d_bias = tf_shell.fast_reduce_sum(dy)
            else:
                d_bias = dy

            d_ws.append(d_bias)

        return d_ws, d_x

    @staticmethod
    def unpack(plaintext_packed_dx):
        batch_size = tf.shape(plaintext_packed_dx)[0] // 2
        return plaintext_packed_dx[0] + plaintext_packed_dx[batch_size]

    def unpacking_funcs(self):
        fs = [ShellDense.unpack]
        if self.use_bias:
            fs.append(ShellDense.unpack)
        return fs
