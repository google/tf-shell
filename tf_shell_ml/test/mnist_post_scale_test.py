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
import numpy as np
import tensorflow as tf
import keras
import tf_shell
from datetime import datetime


log_slots = 11
slots = 2**log_slots
fxp_num_bits = 8  # number of fractional bits.

context = tf_shell.create_context64(
    log_n=log_slots,
    main_moduli=[8556589057, 8388812801],
    aux_moduli=[34359709697],
    plaintext_modulus=40961,
    noise_variance=8,
    seed="",
)
key = tf_shell.create_key64(context)

# Set the batch size to be half the number of slots. This is the maximum
# batch size that can be used with the current implementation of tf-shell
# due to the galois-based ciphertext rotations used in the reduce_sum operations
# which only support rotations of up to half the number of slots.
batch_size = 2**log_slots
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

mnist_layers = [
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="sigmoid"),
]

model = keras.Sequential(mnist_layers)
model.compile(
    optimizer="adam",
    metrics=["accuracy"],
)


def plaintext_train_step(x, y):
    """One step of training with using the "post scale" approach.

    High level idea:
    For each output class, backward pass to compute the gradient.  Now we have a
    _vector_ of model updates for one sample. The real gradient update for the
    sample is a linear combination of the vector of weight updates whose scale
    is determined by dJ_dyhat (the derivative of the loss with respect to the
    predicted output yhat). Effectively, we have factored out dJ_dyhat from the
    gradient. Separating out dJ_dyhat allows us to scale the weight updates
    easily when the label is secret and the gradient must be computed under
    encryption / multiparty computation because the multiplicative depth of the
    computation is 1, however the number of multiplications required now depends
    on the model size AND the number of output classes. In contrast, standard
    backpropagation only requires multiplications proportional to the model size,
    howver the multiplicative depth is proportional to the model depth..
    """

    # First, a quick note about TensorFlow's gradient tape.jacobian().
    # The jacobian tape returns a list of lists of tensors. The outer list is
    # indexed by the layers of the model. Each tensor includes the gradients for
    # all samples in the batch and all output classes. In summary, the
    # indexing looks like:
    #
    # layers list x (batch size x num output classes x weights) matrix
    #
    # Note grads cannot be reduced by any of the above dimensions (e.g.
    # aggregated over a batch by using the tape.gradient function) because
    # each grad's Weight tensor slice needs to be scaled independently.
    #
    # Also note, using tape.gradient to individually compute the gradients for
    # each sample and each output class takes ~10x longer than tape.jacobian.
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # forward pass
    grads = tape.jacobian(
        y_pred, model.trainable_variables
    )  # dy_pred_j/dW_sample_class

    # Compute y_pred - y. The shape is batch_size x num output classes.
    scalars = y_pred - y  # dJ/dy_pred

    # Scale each gradient (layer by layer) by the loss and then aggregating the
    # gradients across the batch and output classes. This is why the
    # approach is called "post scale", because the scaling happens after all
    # possible gradients have already been computed.
    dp_grads = []
    for l, layer_grad_full in enumerate(grads):
        for i in range(layer_grad_full.shape[0]):  # by batch
            for j in range(layer_grad_full.shape[1]):  # by output class
                if i == 0 and j == 0:
                    dp_grads.append(layer_grad_full[i][j] * scalars[i][j])
                else:
                    dp_grads[-1] += (
                        layer_grad_full[i][j] * scalars[i][j]
                    )  # dy_pred_j/dW * dJ/dy_pred = dJ/dW

    # Update weights
    model.optimizer.apply_gradients(zip(dp_grads, model.trainable_variables))

    # Update metrics (includes the metric that tracks the loss)
    model.compiled_metrics.update_state(y, y_pred)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in model.metrics}


def train_step(x, y):
    """One step of training with using the "post scale" approach.

    The high level idea is the same as in plaintext, however in this case
    the gradients are scaled under encryption.
    """

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # forward pass
    grads = tape.jacobian(y_pred, model.trainable_variables)
    # ^  layers list x (batch size x num output classes x weights) matrix
    # dy_pred_j/dW_sample_class

    # Compute y_pred - y under encryption.
    scalars = y_pred - y  # dJ/dy_pred
    # ^  batch_size x num output classes.

    # Scale each gradient. Since 'scalars' may be a vector of ciphertexts, this
    # requires multiplying plaintext gradient for the specific layer (2d) by the
    # ciphertext (scalar). To do so efficiently under encryption requires
    # flattening and packing the weights, as shown below.
    ps_grads = []
    for layer_grad_full in grads:
        # Remember the original shape of the gradient in order to unpack them
        # after the multiplication so they can be applied to the model.
        batch_sz = layer_grad_full.shape[0]
        num_output_classes = layer_grad_full.shape[1]
        grad_shape = layer_grad_full.shape[2:]

        packable_grad = tf.reshape(layer_grad_full, [batch_sz, num_output_classes, -1])
        # ^  batch_size x num output classes x flattened weights

        if isinstance(scalars, tf_shell.ShellTensor64):
            raise NotImplementedError("Not implemented for shell tensors")
            tiled_scalars = tf_shell.tile(
                tf_shell.expand_dims(scalars, axis=-1),
                [1, 1, packable_grad.shape[-1]],
            )
        else:
            tiled_scalars = tf.tile(
                tf.expand_dims(scalars, axis=-1), [1, 1, packable_grad.shape[-1]]
            )
        # ^ batch_size x num output classes x flattened weights

        # Scale the gradient precursors.
        scaled_grad = packable_grad * tiled_scalars
        # ^ dy_pred/dW * dJ/dy_pred = dJ/dW

        # Sum over the output classes.
        if isinstance(scaled_grad, tf_shell.ShellTensor64):
            scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=1)
        else:
            scaled_grad = tf.reduce_sum(scaled_grad, axis=1)
        # scaled_grad is shape
        # ^  batch_size x 1 x flattened weights

        # TODO Clip the gradient, add DP noise, and aggregate over the batch.

        # Decrypt and reshape to remove the '1' dimension in the middle.
        if isinstance(scaled_grad, tf_shell.ShellTensor64):
            scaled_grad = tf_shell.to_tensorflow(scaled_grad, key)
        plaintext_grad = tf.reshape(scaled_grad, [batch_sz] + grad_shape)

        ps_grads.append(plaintext_grad)

    return ps_grads


class TestPlaintextPostScale(unittest.TestCase):
    def test_mnist_shell_post_scale(self):
        start_time = time.time()

        (x_batch, y_batch) = next(iter(train_dataset))

        print(f"Starting stamp: {time.time() - start_time}")

        # Plaintext
        ps_grads = train_step(x_batch, y_batch)

        # Encrypted
        enc_y_batch = tf_shell.to_encrypted(y_batch, key, context)

        shell_ps_grads = train_step(x_batch, enc_y_batch)

        # Compare the gradients.
        self.assertAllClose(
            ps_grads,
            shell_ps_grads,
            atol=slots * 2.0 ** (-fxp_num_bits),
        )

        print(f"Total plaintext training time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    unittest.main()
