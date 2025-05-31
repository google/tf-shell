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
    def _test_model(
        self,
        disable_encryption,
        disable_masking,
        disable_noise,
        clipping_threshold,
        cache,
    ):
        # Prepare the dataset.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
        x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
        y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

        # Clip dataset images to limit memory usage. The model accuracy will be
        # bad but this test only measures functionality.
        x_train, x_test = x_train[:, :350], x_test[:, :350]

        labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        labels_dataset = labels_dataset.batch(2**10)

        features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        features_dataset = features_dataset.batch(2**10)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

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
            backprop_context_fn=lambda read_from_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=23,
                scaling_factor=4,
                noise_offset_log2=14,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            noise_context_fn=lambda read_from_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=25,
                scaling_factor=1,
                noise_offset_log2=0,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            cache_path=cache,
            disable_he_backprop_INSECURE=disable_encryption,
            disable_masking_INSECURE=disable_masking,
            simple_noise_INSECURE=disable_noise,
            simple_noise_clip_threshold=clipping_threshold,
            check_overflow_INSECURE=True,
        )

        m.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        m.build([None, 350])
        m.summary()

        history = m.fit(
            features_dataset,
            labels_dataset,
            steps_per_epoch=4,
            epochs=1,
            verbose=2,
            validation_data=val_dataset,
        )

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.25)

    def test_model(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            # Perform full encrypted test to populate cache.
            self._test_model(False, False, False, None, cache_dir)
            self._test_model(True, False, False, None, cache_dir)
            self._test_model(False, True, False, None, cache_dir)
            self._test_model(False, False, True, None, cache_dir)
            self._test_model(True, True, True, None, cache_dir)
            self._test_model(True, True, True, 1.0, cache_dir)


if __name__ == "__main__":
    unittest.main()
