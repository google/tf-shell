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
# import unittest
import os
import tensorflow as tf
import tempfile

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"


class TestDistribModel(tf.test.TestCase):
    job_name = None

    def test_model(self):
        print(f"Job name: {self.job_name}")
        # flush stdout
        import sys

        sys.stdout.flush()

        cluster = tf.train.ClusterSpec(
            {
                f"{features_party_job}": ["localhost:2223"],
                f"{labels_party_job}": ["localhost:2224"],
            }
        )

        server = tf.distribute.Server(
            cluster,
            job_name=self.job_name,
            task_index=0,
        )

        tf.config.experimental_connect_to_cluster(cluster)

        # Register the tf-shell ops.
        import tf_shell

        if self.job_name == labels_party_job:
            print(f"{self.job_name} server started.", flush=True)
            server.join()
            return

        import keras
        import tf_shell_ml
        import numpy as np

        # Prepare the dataset. (Note this must be after forking)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
        x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
        y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

        # Clip dataset images to limit memory usage. The model accuracy will be
        # bad but this test only measures functionality.
        x_train, x_test = x_train[:, :250], x_test[:, :250]

        # Set a seed for shuffling both features and labels the same way.
        seed = 42

        with tf.device(labels_party_dev):
            labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
            labels_dataset = labels_dataset.batch(2**10)

        with tf.device(features_party_dev):
            features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
            features_dataset = features_dataset.batch(2**10)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            val_dataset = val_dataset.batch(32)

            cache_dir = tempfile.TemporaryDirectory()
            cache = cache_dir.name

            m = tf_shell_ml.PostScaleSequential(
                [
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ],
                lambda read_from_cache: tf_shell.create_autocontext64(
                    log2_cleartext_sz=23,
                    scaling_factor=32,
                    noise_offset_log2=14,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                ),
                lambda read_from_cache: tf_shell.create_autocontext64(
                    log2_cleartext_sz=24,
                    scaling_factor=1,
                    noise_offset_log2=0,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                ),
                labels_party_dev=labels_party_dev,
                features_party_dev=features_party_dev,
                cache_path=cache,
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

        cache_dir.cleanup()

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.25)


if __name__ == "__main__":
    labels_pid = os.fork()
    if labels_pid == 0:  # child process
        TestDistribModel.job_name = labels_party_job
        tf.test.main()
        os._exit(0)

    TestDistribModel.job_name = features_party_job
    tf.test.main()

    os.waitpid(labels_pid, 0)
    print("Both parties finished.")
