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
    For each output class, forward pass to compute the gradient.  Now we have a
    _vector_ of model updates for one sample. The real gradient update for the
    sample is a linear combination of the vector of weight updates whose scale
    is determined by dJ_dypred (the derivative of the loss with respect to the
    predicted output y). Separating out dJ_dypred allows us to scale the weight
    updates easily under encryption / multiparty computation because the
    multiplicative depth of the computation is 1, however it requires
    many more multiplications compared to standard backpropagation.
    """

    # The jacobian tape returns a list of lists of tensors. The outer list is
    # indexed by the layers of the model. Each tensor includes the gradients for
    # all samples in the minibatch and all output classes. In summary, the
    # indexing looks like:
    #
    # Layers x (MiniBatchSz x NumOutputClass x Weights)
    #
    # Note grads cannot be reduced by any of the above dimensions (e.g.
    # aggregated over a minibatch by using the tape.gradient function) because
    # each grad's Weight tensor slice needs to be scaled independently.
    #
    # Also note, using the tape.gradient function to individually compute the
    # gradients for each sample and each output class takes ~10x longer than
    # tape.jacobian.
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # forward pass
    grads = tape.jacobian(
        y_pred, model.trainable_variables
    )  # dy_pred_j/dW_sample_class

    # Compute (y_pred - y) which in theory could be hidden by MPC/HE.
    scalars = y_pred - y  # dJ/dy_pred

    # Scale each gradient (layer by layer) by the plaintext loss gradient
    # and then reduce the gradients across the minibatch and output classes.
    # This is why the approach is called "post scale", because the scaling
    # happens after all gradients are computed.
    dp_grads = []
    for l, layer_grad_full in enumerate(grads):
        dp_grads.append(tf.zeros(layer_grad_full.shape[2:]))

        for i in range(layer_grad_full.shape[0]):
            for j in range(layer_grad_full.shape[1]):
                dp_grads[-1] += (
                    layer_grad_full[i][j] * scalars[i][j]
                )  # dy_pred_j/dW * dJ/dy_pred = dJ/dW

    # Update weights
    model.optimizer.apply_gradients(zip(dp_grads, model.trainable_variables))

    # Update metrics (includes the metric that tracks the loss)
    model.compiled_metrics.update_state(y, y_pred)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in model.metrics}


class TestPlaintextPostScale(unittest.TestCase):
    def test_mnist_plaintext_post_scale(self):
        epochs = 1
        start_time = time.time()

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # Log every 2 batches.
                if step % 2 == 0:
                    print(
                        f"Epoch: {epoch}, Batch: {step} / {len(train_dataset)}, Stamp: {time.time() - start_time}"
                    )

                train_step(x_batch_train, y_batch_train)

                if step == stop_after_n_batches:
                    break

        print(f"Total plaintext training time: {time.time() - start_time} seconds")

        # If necessary, the training can be checked by uncommenting the following.
        # loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
        # self.assertLess(loss, 0.3)
        # self.assertGreater(accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
