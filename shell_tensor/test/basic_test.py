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
import shell_tensor


class TestShellTensor(tf.test.TestCase):
    def get_context():
        ct_params = shell_tensor.shell.ContextParams64(
            modulus=shell_tensor.shell.kModulus59, log_n=10, log_t=16, variance=8
        )
        context_tensor = shell_tensor.create_context64(ct_params)
        return context_tensor

    def test_create_shell_tensor(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform([1024, 3], dtype=tf.int32, maxval=100)
        shelltensor = shell_tensor.to_shell_tensor(context, tftensor)
        tftensor_out = shell_tensor.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor_out, tftensor)

    def test_shape(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform([1024, 3], dtype=tf.int32, maxval=100)
        shelltensor = shell_tensor.to_shell_tensor(context, tftensor)
        self.assertAllClose(tftensor.shape, shelltensor.shape)
        self.assertAllClose(tftensor.shape, shelltensor.shape)

        tftensor_out = shell_tensor.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor.shape, tftensor_out.shape)

    def test_create_shell_tensor_mdim(self):
        context = TestShellTensor.get_context()
        tftensor = tf.random.uniform([1024, 3, 5, 6], dtype=tf.int32, maxval=100)
        shelltensor = shell_tensor.to_shell_tensor(context, tftensor)
        tftensor_out = shell_tensor.from_shell_tensor(shelltensor)
        self.assertAllClose(tftensor_out, tftensor)

    def test_encrypt(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)
        tftensor = tf.random.uniform([1024, 3, 2, 12], dtype=tf.int32, maxval=100)

        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(prng, key)
        tftensor_out = enc.get_decrypted(key)
        self.assertAllClose(tftensor_out, tftensor)


if __name__ == "__main__":
    tf.test.main()
