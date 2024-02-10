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
            noise_variance=8,
            seed="",
        )

    def test_create_shell_tensor_positive(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform(
            [TestShellTensor.slots, 3], dtype=tf.int32, maxval=100
        )
        shelltensor = tf_shell.to_shell_tensor(context, tftensor)
        tftensor_out = tf_shell.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor_out, tftensor)

    def test_create_shell_tensor_negative(self):
        context = TestShellTensor.get_context()
        tftensor = (
            tf.random.uniform([TestShellTensor.slots, 3], dtype=tf.int32, maxval=100)
            - 50
        )
        shelltensor = tf_shell.to_shell_tensor(context, tftensor)
        tftensor_out = tf_shell.from_shell_tensor(shelltensor)
        print(f"tftensor dtype {tftensor.dtype}")
        self.assertAllClose(tftensor_out, tftensor)

    def test_shape(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform(
            [TestShellTensor.slots, 3], dtype=tf.int32, maxval=100
        )
        shelltensor = tf_shell.to_shell_tensor(context, tftensor)
        self.assertAllClose(tftensor.shape, shelltensor.shape)
        self.assertAllClose(tftensor.shape, shelltensor.shape)

        tftensor_out = tf_shell.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor.shape, tftensor_out.shape)

    def test_create_shell_tensor_multi_dim(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform(
            [TestShellTensor.slots, 3, 5, 6], dtype=tf.int32, maxval=100
        )
        shelltensor = tf_shell.to_shell_tensor(context, tftensor)
        tftensor_out = tf_shell.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor_out, tftensor)

    def test_encrypt_decrypt_positive(self):
        context = TestShellTensor.get_context()
        key = tf_shell.create_key64(context)
        tftensor = tf.random.uniform(
            [TestShellTensor.slots, 3, 2, 12], dtype=tf.int32, maxval=100
        )

        s = tf_shell.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)
        tftensor_out = enc.get_decrypted(key)
        self.assertAllClose(tftensor_out, tftensor)

    def test_encrypt_decrypt_negative(self):
        context = TestShellTensor.get_context()
        key = tf_shell.create_key64(context)
        tftensor = (
            tf.random.uniform(
                [TestShellTensor.slots, 3, 2, 12], dtype=tf.int32, maxval=100
            )
            - 50
        )

        s = tf_shell.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)
        tftensor_out = enc.get_decrypted(key)
        self.assertAllClose(tftensor_out, tftensor)


if __name__ == "__main__":
    tf.test.main()
