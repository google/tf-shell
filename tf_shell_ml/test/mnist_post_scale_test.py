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
import numpy as np
import tensorflow as tf
import keras
import tf_shell

# Num plaintext bits: 19, noise bits: 40
# Max representable value: 61895
context = tf_shell.create_context64(
    log_n=11,
    main_moduli=[576460752303439873],
    plaintext_modulus=557057,
    scaling_factor=3,
    mul_depth_supported=1,
)
# 121 bits of security according to lattice estimator primal_bdd.
# Runtime 95 seconds (46 ms/example).

key = tf_shell.create_key64(context)
rotation_key = tf_shell.create_rotation_key64(context, key)

# Set the batch size to be half the number of slots. This is the maximum
# batch size that can be used with the current implementation of tf-shell
# due to the galois-based ciphertext rotations used in the reduce_sum operations
# which only support rotations of up to half the number of slots.
batch_size = context.num_slots

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


@tf.function
def train_step(x, y):
    """One step of training with using the "post scale" approach.

    High level idea:
    For each output class, backprop to compute the gradient but exclude the loss
    function. Now we have a _vector_ of model updates for one sample. The real
    gradient update for the sample is a linear combination of the vector of
    weight updates whose scale is determined by dJ_dyhat (the derivative of the
    loss with respect to the predicted output yhat). Effectively, we have
    factored out dJ_dyhat from the gradient. Separating out dJ_dyhat allows us
    to scale the weight updates easily when the label is secret and the gradient
    must be computed under encryption / multiparty computation because the
    multiplicative depth of the computation is 1, however the number of
    multiplications required now depends on the model size AND the number of
    output classes. In contrast, standard backpropagation only requires
    multiplications proportional to the model size, howver the multiplicative
    depth is proportional to the model depth.
    """

    # Unset the activation function for the last layer so it is not used in
    # computing the gradient. The effect of the last layer activation function
    # is factored out of the gradient computation and accounted for below.
    model.layers[-1].activation = tf.keras.activations.linear

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # forward pass
    grads = tape.jacobian(y_pred, model.trainable_variables)
    # ^  layers list x (batch size x num output classes x weights) matrix
    # dy_pred_j/dW_sample_class

    # Reset the activation function for the last layer and compute the real
    # prediction.
    model.layers[-1].activation = tf.keras.activations.sigmoid
    y_pred = model(x, training=False)

    # Compute y_pred - y (where y may be encrypted).
    # scalars = y_pred - y  # dJ/dy_pred
    scalars = y.__rsub__(y_pred)  # dJ/dy_pred
    # ^  batch_size x num output classes.

    # Expand the last dim so that the subsequent multiplications are
    # broadcasted.
    scalars = tf_shell.expand_dims(scalars, axis=-1)
    # ^ batch_size x num output classes x 1

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

        # Scale the gradient precursors.
        scaled_grad = scalars * packable_grad
        # ^  dJ/dW = dJ/dy_pred * dy_pred/dW

        # Sum over the output classes.
        scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=1)
        # ^  batch_size x flattened weights

        # In the real world, this approach would also likely require clipping
        # the gradient, and adding DP noise.

        # Reshape to unflatten the weights.
        scaled_grad = tf_shell.reshape(scaled_grad, [batch_sz] + grad_shape)
        # ^  batch_size x weights

        # Sum over the batch.
        scaled_grad = tf_shell.reduce_sum(
            scaled_grad, axis=0, rotation_key=rotation_key
        )
        # ^  batch_size x flattened weights

        # Decrypt.
        if isinstance(scaled_grad, tf_shell.ShellTensor64):
            scaled_grad = tf_shell.to_tensorflow(scaled_grad, key)[0]

        ps_grads.append(scaled_grad)

    return ps_grads


class TestPlaintextPostScale(tf.test.TestCase):
    def test_mnist_post_scale_eager(self):
        tf.config.run_functions_eagerly(True)

        (x_batch, y_batch) = next(iter(train_dataset))

        # Plaintext
        ps_grads = train_step(x_batch, y_batch)

        # Encrypted
        enc_y_batch = tf_shell.to_encrypted(y_batch, key, context)
        shell_ps_grads = train_step(x_batch, enc_y_batch)

        # Compare the gradients.
        self.assertAllClose(
            ps_grads,
            shell_ps_grads,
            atol=1 / context.scaling_factor * context.num_slots,
        )

class TestPlaintextPostScale(tf.test.TestCase):
    def test_mnist_post_scale_autograph(self):
        tf.config.run_functions_eagerly(False)

        (x_batch, y_batch) = next(iter(train_dataset))
        
        # Plaintext
        ps_grads = train_step(x_batch, y_batch)

        # With autograph on (eagerly off), the tf.function trace cannot be
        # reused between plaintext and encrypted calls. Reset the graph
        # between plaintext and encrypted train_step() calls.
        tf.keras.backend.clear_session()

        # Encrypted
        enc_y_batch = tf_shell.to_encrypted(y_batch, key, context)
        shell_ps_grads = train_step(x_batch, enc_y_batch)

        # Compare the gradients.
        self.assertAllClose(
            ps_grads,
            shell_ps_grads,
            atol=1 / context.scaling_factor * context.num_slots,
        )


if __name__ == "__main__":
    unittest.main()
