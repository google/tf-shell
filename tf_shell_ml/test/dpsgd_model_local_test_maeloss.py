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
import test_models


class TestModel(tf.test.TestCase):
    def _test_model(self, disable_encryption, disable_masking, disable_noise, cache):
        crop_by = 10
        features_dataset, labels_dataset, val_dataset = test_models.MNIST_datasets(
            crop_by=crop_by
        )

        # Create the model with no activation on the last layer.
        input_img = tf.keras.Input(shape=(28 - crop_by, 28 - crop_by, 1))
        x = tf_shell_ml.Flatten()(input_img)
        x = tf_shell_ml.ShellDense(64, activation="relu")(x)
        x = tf_shell_ml.ShellDense(10)(x)  # No activation

        m = tf_shell_ml.DpSgdModel(
            inputs=input_img,
            outputs=x,
            backprop_context_fn=lambda read_from_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=23,
                scaling_factor=4,
                noise_offset_log2=16,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            noise_context_fn=lambda read_from_cache: tf_shell.create_autocontext64(
                log2_cleartext_sz=25,
                scaling_factor=1,
                noise_offset_log2=48,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            cache_path=cache,
            disable_he_backprop_INSECURE=disable_encryption,
            disable_masking_INSECURE=disable_masking,
            simple_noise_INSECURE=disable_noise,
            check_overflow_INSECURE=True,
        )

        m.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

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
            self._test_model(False, True, False, cache_dir)
            self._test_model(True, True, False, cache_dir)
            self._test_model(True, True, True, cache_dir)


if __name__ == "__main__":
    unittest.main()
