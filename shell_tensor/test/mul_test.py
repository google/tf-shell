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
            modulus=shell_tensor.shell.kModulus59,
            log_n=10,
            log_t=16,
            variance=2,  # Too low for prod. Okay for test.
        )
        context_tensor = shell_tensor.create_context64(ct_params)
        return context_tensor

    def test_ct_ct_mul(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(prng, key)
        eb = sb.get_encrypted(prng, key)

        ec = ea * eb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

    def test_ct_pt_mul(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(prng, key)

        ec = ea * sb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

        ed = sb * ea
        self.assertAllClose(a * b, ed.get_decrypted(key))

    def test_ct_tf_mul(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        ea = sa.get_encrypted(prng, key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(key))

    def test_pt_pt_mul(self):
        context = TestShellTensor.get_context()

        a = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([1024, 2, 3, 4], dtype=tf.int32, maxval=10)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)

        sc = sa * sb
        self.assertAllClose(a * b, shell_tensor.from_shell_tensor(sc))

    def test_ct_tf_matmul(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([1024, 5], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([5, 7], dtype=tf.int32, maxval=10)
        ea = shell_tensor.to_shell_tensor(context, a).get_encrypted(prng, key)

        ec = shell_tensor.matmul(ea, b)
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(tf.matmul(a, b), ec.get_decrypted(key))

    def test_tf_ct_matmul(self):
        context = TestShellTensor.get_context()
        prng = shell_tensor.create_prng()
        key = shell_tensor.create_key64(context, prng)

        a = tf.random.uniform([5, 1024], dtype=tf.int32, maxval=10)
        b = tf.random.uniform([1024, 7], dtype=tf.int32, maxval=10)
        eb = shell_tensor.to_shell_tensor(context, b).get_encrypted(prng, key)

        ec = shell_tensor.matmul(a, eb, prng, key)
        self.assertAllClose(b, eb.get_decrypted(key))

        check_c = tf.matmul(a, b)
        check_c = tf.expand_dims(check_c, axis=0)
        check_c = tf.repeat(check_c, repeats=[1024], axis=0)
        self.assertAllClose(check_c, ec.get_decrypted(key))


if __name__ == "__main__":
    tf.test.main()
