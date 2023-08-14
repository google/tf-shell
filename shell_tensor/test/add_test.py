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


plaintext_dtype = tf.int32


class TestShellTensor(tf.test.TestCase):
    def get_context():
        ct_params = shell_tensor.shell.ContextParams64(
            modulus=shell_tensor.shell.kModulus59, log_n=10, log_t=16, variance=8
        )
        context_tensor = shell_tensor.create_context64(ct_params)
        return context_tensor

    def test_neg(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        nsa = -sa
        self.assertAllClose(-a, shell_tensor.from_shell_tensor(nsa))

        ea = sa.get_encrypted(prng, key)
        nea = -ea
        self.assertAllClose(-a, nea.get_decrypted(key))

        self.assertAllClose(a, ea.get_decrypted(key))

    def test_ct_ct_add(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(prng, key)
        eb = sb.get_encrypted(prng, key)

        ec = ea + eb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = ea - eb
        self.assertAllClose(a - b, ed.get_decrypted(key))

    def test_ct_pt_add(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 1], dtype=plaintext_dtype, maxval=10)
        b = tf.random.uniform([1024, 1], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(prng, key)

        ec = ea + sb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = sb + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        ee = ea - sb
        self.assertAllClose(a - b, ee.get_decrypted(key))

        ef = sb - ea
        self.assertAllClose(b - a, ef.get_decrypted(key))

    def test_ct_tf_add(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        ea = sa.get_encrypted(prng, key)

        ec = ea + b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = b + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        ee = ea - b
        self.assertAllClose(a - b, ee.get_decrypted(key))

        ef = b - ea
        self.assertAllClose(b - a, ef.get_decrypted(key))

    def test_pt_pt_add(self):
        context = TestShellTensor.get_context()

        a = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)

        sc = sa + sb
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = sa - sb
        self.assertAllClose(a - b, shell_tensor.from_shell_tensor(sd))

    def test_pt_tf_add(self):
        context = TestShellTensor.get_context()

        a = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=plaintext_dtype, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)

        sc = sa + b
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = b + sa
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sd))

        se = sa - b
        self.assertAllClose(a - b, shell_tensor.from_shell_tensor(se))

        sf = b + sa
        self.assertAllClose(b + a, shell_tensor.from_shell_tensor(sf))


if __name__ == "__main__":
    tf.test.main()
