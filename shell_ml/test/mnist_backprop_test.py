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
import unittest
import time
from datetime import datetime
import tensorflow as tf
import keras
import numpy as np
import shell_tensor
import shell_ml

plaintext_dtype = tf.float32
fxp_num_bits = 5  # number of fractional bits.


# Shell setup.
log_slots = 11
slots = 2**log_slots

# Num plaintext bits: 27, noise bits: 65, num rns moduli: 2
context = shell_tensor.create_context64(
    log_n=11,
    main_moduli=[140737488486401, 140737488498689],
    aux_moduli=[],
    plaintext_modulus=134246401,
    noise_variance=8,
    seed="",
)
key = shell_tensor.create_key64(context)
rotation_key = shell_tensor.create_rotation_key64(context, key)

# Training setup.
epochs = 1
batch_size = slots
stop_after_n_batches = 1

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)


# Create the layers
hidden_layer = shell_ml.ShellDense(
    64,
    activation=shell_ml.relu,
    activation_deriv=shell_ml.relu_deriv,
    fxp_fractional_bits=fxp_num_bits,
    weight_dtype=plaintext_dtype,
)
output_layer = shell_ml.ShellDense(
    10,
    activation=shell_ml.sigmoid,
    # activation_deriv=shell_ml.sigmoid_deriv,
    fxp_fractional_bits=fxp_num_bits,
    weight_dtype=plaintext_dtype,
)

# Call the layers once to create the weights.
y1 = hidden_layer(tf.zeros((batch_size, 784)))
y2 = output_layer(y1)

loss_fn = shell_ml.CategoricalCrossentropy()
optimizer = shell_ml.Adam()
optimizer.compile([hidden_layer.weights, output_layer.weights])


def train_step(x, y):
    # Forward pass always in plaintext.
    y_1 = hidden_layer(x)
    y_pred = output_layer(y_1)
    # loss = loss_fn(y, y_pred) # this is expensive and not needed for training

    # Backward pass.
    dJ_dy_pred = loss_fn.grad(y, y_pred)

    (dJ_dw1, dJ_dx1) = output_layer.backward(
        dJ_dy_pred, rotation_key, is_first_layer=False
    )

    (dJ_dw0, dJ_dx0_unused) = hidden_layer.backward(
        dJ_dx1, rotation_key, is_first_layer=True
    )

    # Only return the weight gradients at [0], not the bias gradients at [1].
    return dJ_dw1[0], dJ_dw0[0]


class TestMNISTBackprop(tf.test.TestCase):
    def test_mnist_plaintext_backprop(self):
        (x_batch, y_batch) = next(iter(train_dataset))

        start_time = time.time()

        # Plaintext backprop splitting the batch in half vertically.
        top_x_batch, bottom_x_batch = tf.split(x_batch, num_or_size_splits=2, axis=0)
        top_y_batch, bottom_y_batch = tf.split(y_batch, num_or_size_splits=2, axis=0)
        top_output_layer_grad, top_hidden_layer_grad = train_step(
            top_x_batch, top_y_batch
        )
        bottom_output_layer_grad, bottom_hidden_layer_grad = train_step(
            bottom_x_batch, bottom_y_batch
        )

        # Stack the top and bottom gradients back together along a new
        # outer dimension.
        output_layer_grad = tf.concat(
            [
                tf.expand_dims(top_output_layer_grad, axis=0),
                tf.expand_dims(bottom_output_layer_grad, axis=0),
            ],
            axis=0,
        )
        hidden_layer_grad = tf.concat(
            [
                tf.expand_dims(top_hidden_layer_grad, axis=0),
                tf.expand_dims(bottom_hidden_layer_grad, axis=0),
            ],
            axis=0,
        )

        # Encrypt y using fixed point representation.
        enc_y_batch = shell_tensor.to_shell_tensor(
            context, y_batch, fxp_fractional_bits=fxp_num_bits
        ).get_encrypted(key)

        # Backprop.
        enc_output_layer_grad, enc_hidden_layer_grad = train_step(x_batch, enc_y_batch)

        # Decrypt the gradients.
        repeated_output_layer_grad = enc_output_layer_grad.get_decrypted(key)
        repeated_hidden_layer_grad = enc_hidden_layer_grad.get_decrypted(key)

        print(f"\tFinished Stamp: {time.time() - start_time}")
        print(f"\tOutput Layer Noise: {enc_output_layer_grad.noise_bits}")
        print(f"\tHidden Layer Noise: {enc_hidden_layer_grad.noise_bits}")
        print(
            f"\tOutput Layer fxp bits: {enc_output_layer_grad.num_fxp_fractional_bits}"
        )
        print(
            f"\tHidden Layer fxp bits: {enc_hidden_layer_grad.num_fxp_fractional_bits}"
        )

        shell_output_layer_grad = tf.concat(
            [
                tf.expand_dims(repeated_output_layer_grad[0, ...], 0),
                tf.expand_dims(repeated_output_layer_grad[slots // 2, ...], 0),
            ],
            axis=0,
        )
        shell_hidden_layer_grad = tf.concat(
            [
                tf.expand_dims(repeated_hidden_layer_grad[0, ...], 0),
                tf.expand_dims(repeated_hidden_layer_grad[slots // 2, ...], 0),
            ],
            axis=0,
        )

        # Compare the gradients.
        self.assertAllClose(
            output_layer_grad,
            shell_output_layer_grad,
            atol=slots * 2.0 ** (-fxp_num_bits),
        )

        self.assertAllClose(
            hidden_layer_grad,
            shell_hidden_layer_grad,
            atol=slots * 2.0 ** (-fxp_num_bits - 2),
        )

        print(f"Total plaintext training time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    unittest.main()
