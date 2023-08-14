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


# Shell setup
def get_context():
    ct_params = shell_tensor.shell.ContextParams64(
        modulus=shell_tensor.shell.kModulus59,
        log_n=10,
        log_t=16,
        variance=0,  # Too low for prod. Okay for test.
    )
    context_tensor = shell_tensor.create_context64(ct_params)
    return context_tensor


context = get_context()
prng = shell_tensor.create_prng()
key = shell_tensor.create_key64(context, prng)


batch_size = 2**10
stop_after_n_batches = 4

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
    64, activation=shell_ml.relu, activation_deriv=shell_ml.relu_deriv
)
output_layer = shell_ml.ShellDense(
    10, activation=shell_ml.sigmoid, activation_deriv=shell_ml.sigmoid_deriv
)

# Call the layers to create the weights
y1 = hidden_layer(tf.zeros((batch_size, 784)))
y2 = output_layer(y1)

loss_fn = shell_ml.CategoricalCrossentropy()
optimizer = shell_ml.Adam()
optimizer.compile([hidden_layer.weights, output_layer.weights])


@tf.function
def train_step(x, y):
    # In practice, input y would be quantized to fixed point before encryption.
    # This is not done here to reduce dependencies on external libraries.
    y = tf.cast(y, tf.int32)

    # Encrypt y
    y = shell_tensor.to_shell_tensor(context, y).get_encrypted(prng, key)

    # Forward pass in plaintext
    y_1 = hidden_layer(x)
    y_pred = output_layer(y_1)
    # loss = loss_fn(y, y_pred) # this is expensive and not needed for training

    # Backward pass under encryption
    dJ_dy_pred = loss_fn.grad(y, y_pred)
    (dJ_dw1, dJ_dx1) = output_layer.backward(dJ_dy_pred, False, prng, key)
    (dJ_dw0, dJ_dx0_unused) = hidden_layer.backward(dJ_dx1, True, prng, key)

    # In practice, the gradients are likely secret and should be aggregated and
    # noised before decryption. Additionally, weight gradients are in a form
    # where Decryption may be more efficient if ciphertexts are first packed.
    dJ_dw1 = dJ_dw1[0].get_decrypted(key)
    dJ_dw0 = dJ_dw0[0].get_decrypted(key)

    # Decrypt and apply the weight gradients. dJ_dw[1] is bias.
    dJ_dw1 = tf.cast(dJ_dw1, tf.float32)
    optimizer.grad_to_weight(output_layer.weights, dJ_dw1)

    dJ_dw0 = tf.cast(dJ_dw0, tf.float32)
    optimizer.grad_to_weight(hidden_layer.weights, dJ_dw0)


class TestPlaintextPostScale(unittest.TestCase):
    def test_mnist_shell_backprop(self):
        epochs = 1
        start_time = time.time()

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # Log every 2 batches.
                if step % 2 == 0:
                    print(
                        f"Epoch: {epoch}, Batch: {step} / {len(train_dataset)}, Time: {time.time() - start_time}"
                    )

                train_step(x_batch_train, y_batch_train)

                if step == stop_after_n_batches:
                    break

        print(f"Total plaintext training time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    unittest.main()
