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
import tensorflow as tf
import keras
import numpy as np
import tf_shell
import tf_shell_ml

# # Num plaintext bits: 32, noise bits: 84
# # Max representable value: 654624
# context = tf_shell.create_context64(
#     log_n=11,
#     main_moduli=[288230376151748609, 144115188076060673],
#     plaintext_modulus=4294991873,
#     scaling_factor=3,
#     mul_depth_supported=3,
#     seed="test_seed",
# )
# 61 bits of security according to lattice estimator primal_bdd.
# Runtime 170 seconds (83ms/example).

# Num plaintext bits: 32, noise bits: 84
# Max representable value: 654624
context = tf_shell.create_context64(
    log_n=12,
    main_moduli=[288230376151760897, 288230376152137729],
    plaintext_modulus=4294991873,
    scaling_factor=3,
    mul_depth_supported=3,
    seed="test_seed",
)
# 120 bits of security according to lattice estimator primal_bdd.
# Runtime 388 seconds (95ms/example).

key = tf_shell.create_key64(context)
rotation_key = tf_shell.create_rotation_key64(context, key)

# Prepare the dataset.
batch_size = context.num_slots
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)


# Create the layers
hidden_layer = tf_shell_ml.ShellDense(
    64,
    activation=tf_shell_ml.relu,
    activation_deriv=tf_shell_ml.relu_deriv,
    is_first_layer=True,
)
output_layer = tf_shell_ml.ShellDense(
    10,
    activation=tf.nn.softmax,
    # Do not set the derivative of the activation function for the last layer
    # in the model. The derivative of the categorical crossentropy loss function
    # times the derivative of a softmax is just y_pred - y (which is much easier
    # to compute than each of them individually). So instead just let the
    # loss function derivative incorporate y_pred - y and let the derivative
    # of this last layer's activation be a no-op.
)

# Call the layers once to create the weights.
y1 = hidden_layer(tf.zeros((batch_size, 784)))
y2 = output_layer(y1)

loss_fn = tf_shell_ml.CategoricalCrossentropy()


@tf.function
def train_step(x, y):
    # Forward pass.
    y_1 = hidden_layer(x)
    y_pred = output_layer(y_1)
    # loss = loss_fn(y, y_pred)  # Expensive and not needed for this test.

    # Backward pass.
    dJ_dy_pred = loss_fn.grad(y, y_pred)

    dJ_dw1, dJ_dx1 = output_layer.backward(dJ_dy_pred, rotation_key)

    # Mod reduce will reduce noise but increase the plaintext error.
    # if isinstance(dJ_dx1, tf_shell.ShellTensor64):
    #     dJ_dx1.get_mod_reduced()

    dJ_dw0, dJ_dx0_unused = hidden_layer.backward(dJ_dx1, rotation_key)

    # Only return the weight gradients at [0], not the bias gradients at [1].
    return dJ_dw1[0], dJ_dw0[0]


class TestMNISTBackprop(tf.test.TestCase):
    def test_mnist_enc_backprop(self):
        (x_batch, y_batch) = next(iter(train_dataset))

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

        # Encrypt y.
        enc_y_batch = tf_shell.to_encrypted(y_batch, key, context)

        # Backprop.
        enc_output_layer_grad, enc_hidden_layer_grad = train_step(x_batch, enc_y_batch)

        # Decrypt the gradients.
        repeated_output_layer_grad = tf_shell.to_tensorflow(enc_output_layer_grad, key)
        repeated_hidden_layer_grad = tf_shell.to_tensorflow(enc_hidden_layer_grad, key)

        print(f"\tOutput Layer Noise: {enc_output_layer_grad.noise_bits}")
        print(f"\tHidden Layer Noise: {enc_hidden_layer_grad.noise_bits}")

        shell_output_layer_grad = tf.concat(
            [
                tf.expand_dims(repeated_output_layer_grad[0, ...], 0),
                tf.expand_dims(
                    repeated_output_layer_grad[context.num_slots // 2, ...], 0
                ),
            ],
            axis=0,
        )
        shell_hidden_layer_grad = tf.concat(
            [
                tf.expand_dims(repeated_hidden_layer_grad[0, ...], 0),
                tf.expand_dims(
                    repeated_hidden_layer_grad[context.num_slots // 2, ...], 0
                ),
            ],
            axis=0,
        )

        # Compare the gradients.
        self.assertAllClose(
            output_layer_grad,
            shell_output_layer_grad,
            atol=1 / context.scaling_factor * context.num_slots,
            rtol=1 / context.scaling_factor * 2,
        )

        self.assertAllClose(
            hidden_layer_grad,
            shell_hidden_layer_grad,
            atol=1 / context.scaling_factor * context.num_slots,
            rtol=1 / context.scaling_factor * 3,
        )


if __name__ == "__main__":
    unittest.main()
