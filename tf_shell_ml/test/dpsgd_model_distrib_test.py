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

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
coordinator_party_job = f"{job_prefix}coordinator"
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
                f"{coordinator_party_job}": ["localhost:2222"],
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

        if self.job_name != coordinator_party_job:
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
        x_train, x_test = x_train[:, :120], x_test[:, :120]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=2**14).batch(4)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

        m = tf_shell_ml.DpSgdSequential(
            [
                tf_shell_ml.ShellDense(
                    64,
                    activation=tf_shell_ml.relu,
                    activation_deriv=tf_shell_ml.relu_deriv,
                    use_fast_reduce_sum=True,
                ),
                tf_shell_ml.ShellDense(
                    10,
                    activation=tf.nn.softmax,
                    use_fast_reduce_sum=True,
                ),
            ],
            # lambda: tf_shell.create_context64(
            #     log_n=12,
            #     main_moduli=[288230376151760897, 288230376152137729],
            #     plaintext_modulus=4294991873,
            #     scaling_factor=3,
            # ),
            lambda: tf_shell.create_autocontext64(
                log2_cleartext_sz=32,
                scaling_factor=3,
                noise_offset_log2=57,
            ),
            True,
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
        )

        m.compile(
            shell_loss=tf_shell_ml.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        history = m.fit(
            train_dataset.take(2**13),
            epochs=1,
            validation_data=val_dataset,
        )


if __name__ == "__main__":
    labels_pid = os.fork()
    if labels_pid == 0:  # child process
        TestDistribModel.job_name = labels_party_job
        tf.test.main()
        os._exit(0)

    features_pid = os.fork()
    if features_pid == 0:  # child process
        TestDistribModel.job_name = features_party_job
        tf.test.main()
        os._exit(0)

    TestDistribModel.job_name = coordinator_party_job
    tf.test.main()

    os.waitpid(labels_pid, 0)
    os.waitpid(features_pid, 0)
    print("Both parties finished.")
