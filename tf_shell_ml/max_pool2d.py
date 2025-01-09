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


# For details see https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
# and https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks
class MaxPool2D(keras.layers.Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="same",  # valid is not supported for encrypted backprop
        is_first_layer=False,
    ):
        super().__init__()
        self.pool_size = pool_size
        if strides is None:
            self.strides = [1, 1]
        elif isinstance(strides, int):
            self.strides = [strides, strides]
        else:
            self.strides = strides
        self.padding_str = padding.upper()
        self.is_first_layer = is_first_layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding_str,
                "is_first_layer": self.is_first_layer,
            }
        )
        return config

    def reset_split_forward_mode(self):
        self._layer_intermediate = []
        self._layer_input_shape = None

    def call(self, inputs, training=False, split_forward_mode=False):
        """Inputs are expected to be in NHWC format, i.e.
        [batch, height, width, channels]
        """
        # TODO: Something is setting `outputs` to inputs in dpsgd model.
        outputs, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            self.pool_size,
            self.strides,
            self.padding_str,
            include_batch_in_index=False,
        )

        if training:
            if split_forward_mode:
                self._layer_intermediate.append(argmax)
                if self._layer_input_shape is None:
                    self._layer_input_shape = inputs.shape.as_list()
                else:
                    # if self._layer_input_shape[1:] != inputs.shape.as_list()[1:]:
                    #     raise ValueError(
                    #         f"Expected input shape {self._layer_input_shape[1:]}, "
                    #         f"got {inputs.shape.as_list()[1:]}"
                    #     )
                    self._layer_input_shape[0] += inputs.shape.as_list()[0]

            else:
                self._layer_intermediate = [argmax]
                self._layer_input_shape = inputs.shape.as_list()

        return outputs

    def backward(self, dy, rotation_key=None):
        """Compute the gradient."""
        indices = tf.concat([tf.identity(z) for z in self._layer_intermediate], axis=0)
        grad_weights = []

        # On the forward pass, inputs may be batched differently than the
        # ciphertext scheme when not in eager mode. Pad them to match the
        # ciphertext scheme.
        if isinstance(dy, tf_shell.ShellTensor64):
            padding = [[0, dy._context.num_slots - indices.shape[0]]] + [
                [0, 0] for _ in range(len(indices.shape) - 1)
            ]
            indices = tf.pad(indices, padding)

        if self.is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            d_x = tf_shell.max_unpool2d(
                dy,
                indices,
                self.pool_size,
                self.strides,
                self.padding_str,
                output_shape=self._layer_input_shape,
            )

        return grad_weights, d_x

    def unpacking_funcs(self):
        return []
