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
import os
import tempfile


class TestModel(tf.test.TestCase):
    def _test_model(self, disable_encryption, disable_masking, disable_noise, cache):
        # Prepare the dataset.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
        x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
        y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=2**10).batch(2**12)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        # Turn on the shell optimizer to use autocontext.
        tf_shell.enable_optimization()

        m = tf_shell_ml.DpSgdSequential(
            [
                tf_shell_ml.ShellDense(
                    64,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                ),
                tf_shell_ml.ShellDense(
                    10,
                    activation=tf.nn.softmax,
                ),
            ],
            backprop_context_fn=lambda: tf_shell.create_autocontext64(
                log2_cleartext_sz=23,
                scaling_factor=32,
                noise_offset_log2=14,
                cache_path=cache,
            ),
            noise_context_fn=lambda: tf_shell.create_autocontext64(
                log2_cleartext_sz=24,
                scaling_factor=1,
                noise_offset_log2=0,
                cache_path=cache,
            ),
            disable_encryption=disable_encryption,
            disable_masking=disable_masking,
            disable_noise=disable_noise,
            cache_path=cache,
            check_overflow_INSECURE=True,
        )

        m.compile(
            shell_loss=tf_shell_ml.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.1),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        m.build([None, 784])
        m.summary()

        history = m.fit(
            train_dataset,
            steps_per_epoch=8,
            epochs=1,
            verbose=2,
            validation_data=val_dataset,
        )

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.30)

    def test_model(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            self._test_model(False, False, False, cache_dir)
            self._test_model(True, False, False, cache_dir)
            self._test_model(False, True, False, cache_dir)
            self._test_model(False, False, True, cache_dir)


if __name__ == "__main__":
    unittest.main()
