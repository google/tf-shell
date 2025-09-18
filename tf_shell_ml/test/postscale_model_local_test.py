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
    def _test_model(
        self,
        disable_encryption,
        disable_masking,
        disable_noise,
        cache,
        use_autocontext=False,
    ):
        crop_by = 12
        features_dataset, labels_dataset, val_dataset = test_models.MNIST_datasets(
            crop_by=crop_by
        )

        inputs, outputs = test_models.MNIST_FF(
            10, inputs=(28 - crop_by, 28 - crop_by, 1)
        )

        def backprop_context_fn(read_from_cache):
            if use_autocontext:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=23,
                    scaling_factor=32,
                    noise_offset_log2=14,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                )
            return tf_shell.create_context64(
                log_n=12,
                main_moduli=[140805821931521, 70437337817089],
                plaintext_modulus=8404993,
                scaling_factor=32,
            )

        def noise_context_fn(read_from_cache):
            if use_autocontext:
                return tf_shell.create_autocontext64(
                    log2_cleartext_sz=24,
                    scaling_factor=1,
                    noise_offset_log2=0,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                )
            return tf_shell.create_context64(
                log_n=12,
                main_moduli=[963482017793, 2477525188609],
                plaintext_modulus=16801793,
                scaling_factor=1,
            )

        m = tf_shell_ml.PostScaleModel(
            inputs=inputs,
            outputs=outputs,
            ubatch_per_batch=2,
            backprop_context_fn=backprop_context_fn,
            noise_context_fn=noise_context_fn,
            cache_path=cache,
            disable_he_backprop_INSECURE=disable_encryption,
            disable_masking_INSECURE=disable_masking,
            simple_noise_INSECURE=disable_noise,
            check_overflow_INSECURE=True,
        )

        m.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        history = m.fit(
            features_dataset,
            labels_dataset,
            steps_per_epoch=4,
            epochs=1,
            verbose=2,
            validation_data=val_dataset,
        )

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.55)

    def test_model(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            eager = False
            tf.config.run_functions_eagerly(eager)
            # Perform full encrypted test to populate cache.
            self._test_model(False, False, False, cache_dir, use_autocontext=not eager)
            self._test_model(False, True, False, cache_dir, use_autocontext=not eager)
            self._test_model(True, True, False, cache_dir, use_autocontext=not eager)
            self._test_model(True, True, True, cache_dir, use_autocontext=not eager)


if __name__ == "__main__":
    unittest.main()
