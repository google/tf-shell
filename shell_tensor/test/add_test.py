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
    plaintext_dtype = tf.int32
    log_slots = 11
    slots = 2**log_slots

    def get_context():
        return shell_tensor.create_context64(
            log_n=TestShellTensor.log_slots,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[34359709697],
            plaintext_modulus=40961,
            noise_variance=8,
            seed="",
        )

    def test_neg(self):
        context = TestShellTensor.get_context()
        key = shell_tensor.create_key64(context)

        a = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        sa = shell_tensor.to_shell_tensor(context, a)
        nsa = -sa
        self.assertAllClose(-a, shell_tensor.from_shell_tensor(nsa))

        ea = sa.get_encrypted(key)
        nea = -ea
        self.assertAllClose(-a, nea.get_decrypted(key))

        self.assertAllClose(a, ea.get_decrypted(key))

    def test_ct_ct_add(self):
        context = TestShellTensor.get_context()
        key = shell_tensor.create_key64(context)

        a = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        b = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(key)
        eb = sb.get_encrypted(key)

        ec = ea + eb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = ea - eb
        self.assertAllClose(a - b, ed.get_decrypted(key))

    def test_ct_pt_add(self):
        context = TestShellTensor.get_context()
        key = shell_tensor.create_key64(context)

        a = tf.random.uniform(
            [TestShellTensor.slots, 1], dtype=TestShellTensor.plaintext_dtype, maxval=10
        )
        b = tf.random.uniform(
            [TestShellTensor.slots, 1], dtype=TestShellTensor.plaintext_dtype, maxval=10
        )
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(key)

        ec = ea + sb
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = sb + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        ee = ea - sb
        self.assertAllClose(a - b, ee.get_decrypted(key))

        ef = sb - ea
        self.assertAllClose(b - a, ef.get_decrypted(key))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_ct_tf_add(self):
        context = TestShellTensor.get_context()
        key = shell_tensor.create_key64(context)

        a = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        b = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        sa = shell_tensor.to_shell_tensor(context, a)
        ea = sa.get_encrypted(key)

        ec = ea + b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = b + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        ee = ea - b
        self.assertAllClose(a - b, ee.get_decrypted(key))

        ef = b - ea
        self.assertAllClose(b - a, ef.get_decrypted(key))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, ea.get_decrypted(key))

    def test_pt_pt_add(self):
        context = TestShellTensor.get_context()

        a = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        b = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)

        sc = sa + sb
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = sa - sb
        self.assertAllClose(a - b, shell_tensor.from_shell_tensor(sd))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_pt_tf_add(self):
        context = TestShellTensor.get_context()

        a = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        b = tf.random.uniform(
            [TestShellTensor.slots, 2, 3, 4],
            dtype=TestShellTensor.plaintext_dtype,
            maxval=10,
        )
        sa = shell_tensor.to_shell_tensor(context, a)

        sc = sa + b
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = b + sa
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sd))

        se = sa - b
        self.assertAllClose(a - b, shell_tensor.from_shell_tensor(se))

        sf = b + sa
        self.assertAllClose(b + a, shell_tensor.from_shell_tensor(sf))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))


if __name__ == "__main__":
    tf.test.main()
