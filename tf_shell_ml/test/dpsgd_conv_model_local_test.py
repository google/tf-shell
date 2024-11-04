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
import tempfile


class TestModel(tf.test.TestCase):
    def _test_model(self, disable_encryption, disable_masking, disable_noise, cache):
        # Prepare the dataset.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(
            x_test, (-1, 28, 28, 1)
        )
        x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
        y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

        labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        labels_dataset = labels_dataset.batch(2**12)

        features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        features_dataset = features_dataset.batch(2**12)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        m = tf_shell_ml.DpSgdSequential(
            [
                # Model from tensorflow-privacy tutorial. The first layer may
                # be skipped and the model still has ~95% accuracy (plaintext).
                # tf_shell_ml.Conv2D(
                #     filters=16,
                #     kernel_size=8,
                #     strides=2,
                #     padding="same",
                #     activation=tf_shell_ml.relu,
                #     activation_deriv=tf_shell_ml.relu_deriv,
                # ),
                # tf_shell_ml.MaxPool2D(
                #     pool_size=(2, 2),
                #     strides=1,
                # ),
                tf_shell_ml.Conv2D(
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    padding="valid",
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                ),
                tf_shell_ml.MaxPool2D(
                    pool_size=(2, 2),
                    strides=1,
                ),
                tf_shell_ml.Flatten(),
                tf_shell_ml.ShellDense(
                    32,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                ),
                tf_shell_ml.ShellDense(
                    10,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=lambda: tf_shell.create_autocontext64(
                log2_cleartext_sz=18,
                scaling_factor=2,
                noise_offset_log2=-20,
                cache_path=cache,
            ),
            noise_context_fn=lambda: tf_shell.create_autocontext64(
                log2_cleartext_sz=36,
                scaling_factor=1,
                noise_offset_log2=35,
                cache_path=cache,
            ),
            disable_encryption=disable_encryption,
            disable_masking=disable_masking,
            disable_noise=disable_noise,
            cache_path=cache,
            # check_overflow_INSECURE=True,
        )

        m.compile(
            shell_loss=tf_shell_ml.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.1),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        m.build([None, 28, 28, 1])
        m.summary()

        history = m.fit(
            features_dataset,
            labels_dataset,
            steps_per_epoch=8,
            epochs=1,
            verbose=2,
            validation_data=val_dataset,
        )

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.25)

    def test_model(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            # Perform full encrypted test to populate cache.
            self._test_model(False, False, False, cache_dir)
            self._test_model(True, False, False, cache_dir)
            self._test_model(False, True, False, cache_dir)
            self._test_model(False, False, True, cache_dir)
            self._test_model(True, True, True, cache_dir)


if __name__ == "__main__":
    unittest.main()
