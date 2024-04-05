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
import tensorflow as tf
import tf_shell

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


class TestShellTensor(tf.test.TestCase):
    log_slots = 11
    slots = 2**log_slots

    def get_context():
        return tf_shell.create_context64(
            log_n=TestShellTensor.log_slots,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[34359709697],
            plaintext_modulus=40961,
        )

    def test_create_shell_tensor_positive(self):
        context = TestShellTensor.get_context()
        tf_tensor = tf.random.uniform(
            [TestShellTensor.slots, 3], dtype=tf.int32, maxval=100
        )
        shell_tensor = tf_shell.to_shell_plaintext(tf_tensor, context)
        tf_tensor_out = tf_shell.to_tensorflow(shell_tensor)
        self.assertAllClose(tf_tensor_out, tf_tensor)

    def test_create_shell_tensor_negative(self):
        context = TestShellTensor.get_context()
        tf_tensor = (
            tf.random.uniform([TestShellTensor.slots, 3], dtype=tf.int32, maxval=100)
            - 50
        )
        shell_tensor = tf_shell.to_shell_plaintext(tf_tensor, context)
        tf_tensor_out = tf_shell.to_tensorflow(shell_tensor)
        print(f"tf_tensor dtype {tf_tensor.dtype}")
        self.assertAllClose(tf_tensor_out, tf_tensor)

    def test_shape(self):
        context = TestShellTensor.get_context()
        tf_tensor = tf.random.uniform(
            [TestShellTensor.slots, 3], dtype=tf.int32, maxval=100
        )
        shell_tensor = tf_shell.to_shell_plaintext(tf_tensor, context)
        self.assertAllClose(tf_tensor.shape, shell_tensor.shape)
        self.assertAllClose(tf_tensor.shape, shell_tensor.shape)

        tf_tensor_out = tf_shell.to_tensorflow(shell_tensor)
        self.assertAllClose(tf_tensor.shape, tf_tensor_out.shape)

    def test_create_shell_tensor_multi_dim(self):
        context = TestShellTensor.get_context()
        tf_tensor = tf.random.uniform(
            [TestShellTensor.slots, 3, 5, 6], dtype=tf.int32, maxval=100
        )
        shell_tensor = tf_shell.to_shell_plaintext(tf_tensor, context)
        tf_tensor_out = tf_shell.to_tensorflow(shell_tensor)
        self.assertAllClose(tf_tensor_out, tf_tensor)

    def test_encrypt_decrypt_positive(self):
        context = TestShellTensor.get_context()
        key = tf_shell.create_key64(context)
        tf_tensor = tf.random.uniform(
            [TestShellTensor.slots, 3, 2, 12], dtype=tf.int32, maxval=100
        )

        enc = tf_shell.to_encrypted(tf_tensor, key, context)
        tf_tensor_out = tf_shell.to_tensorflow(enc, key)
        self.assertAllClose(tf_tensor_out, tf_tensor)

    def test_encrypt_decrypt_negative(self):
        context = TestShellTensor.get_context()
        key = tf_shell.create_key64(context)
        tf_tensor = (
            tf.random.uniform(
                [TestShellTensor.slots, 3, 2, 12], dtype=tf.int32, maxval=100
            )
            - 50
        )

        enc = tf_shell.to_encrypted(tf_tensor, key, context)
        tf_tensor_out = tf_shell.to_tensorflow(enc, key)
        self.assertAllClose(tf_tensor_out, tf_tensor)


if __name__ == "__main__":
    tf.test.main()
