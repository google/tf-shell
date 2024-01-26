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
import test_utils


class TestShellTensor(tf.test.TestCase):
    def _test_neg(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs zero additions, just negation.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 0
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        tf.cast(a, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a)
        nsa = -sa
        self.assertAllClose(-a, shell_tensor.from_shell_tensor(nsa))

        ea = sa.get_encrypted(key)
        nea = -ea
        self.assertAllClose(-a, nea.get_decrypted(key))

        self.assertAllClose(a, ea.get_decrypted(key))

    def test_neg(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    if test_dtype.is_unsigned:
                        # Negating an unsigned value is undefined.
                        continue
                    with self.subTest(
                        "neg with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_neg(test_context, test_dtype, frac_bits)

    def _test_ct_ct_add(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one addition.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)

        ea = shell_tensor.to_shell_tensor(context, a).get_encrypted(key)
        eb = shell_tensor.to_shell_tensor(context, b).get_encrypted(key)

        ec = ea + eb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(context, a + max_val).get_encrypted(key)
            ed = eaa - eb
            self.assertAllClose(a + max_val - b, ed.get_decrypted(key))
        else:
            ed = ea - eb
            self.assertAllClose(a - b, ed.get_decrypted(key))

    def test_ct_ct_add(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_ct_add with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_ct_add(test_context, test_dtype, frac_bits)

    def _test_ct_pt_add(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one addition.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)
        ea = sa.get_encrypted(key)

        ec = ea + sb
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = sb + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(context, a + max_val).get_encrypted(key)
            ee = eaa - sb
            self.assertAllClose(a + max_val - b, ee.get_decrypted(key))

            sbb = shell_tensor.to_shell_tensor(context, b + max_val)
            ef = sbb - ea
            self.assertAllClose(b + max_val - a, ef.get_decrypted(key))
        else:
            ee = ea - sb
            self.assertAllClose(a - b, ee.get_decrypted(key))

            ef = sb - ea
            self.assertAllClose(b - a, ef.get_decrypted(key))

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_ct_pt_add(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_pt_add with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_pt_add(test_context, test_dtype, frac_bits)

    def _test_ct_tf_add(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one addition.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a)
        ea = sa.get_encrypted(key)

        ec = ea + b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a + b, ec.get_decrypted(key))

        ed = b + ea
        self.assertAllClose(a + b, ed.get_decrypted(key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(context, a + max_val).get_encrypted(key)
            ee = eaa - b
            self.assertAllClose(a + max_val - b, ee.get_decrypted(key))

            bb = b + max_val
            ef = bb - ea
            self.assertAllClose(bb - a, ef.get_decrypted(key))
        else:
            ee = ea - b
            self.assertAllClose(a - b, ee.get_decrypted(key))

            ef = b - ea
            self.assertAllClose(b - a, ef.get_decrypted(key))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, ea.get_decrypted(key))

    def test_ct_tf_add(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_tf_add with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_tf_add(test_context, test_dtype, frac_bits)

    def _test_pt_pt_add(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context

        # This test performs one addition.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a)
        sb = shell_tensor.to_shell_tensor(context, b)

        sc = sa + sb
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            saa = shell_tensor.to_shell_tensor(context, a + max_val)
            ee = saa - sb
            self.assertAllClose(a + max_val - b, shell_tensor.from_shell_tensor(ee))
        else:
            sd = sa - sb
            self.assertAllClose(a - b, shell_tensor.from_shell_tensor(sd))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_pt_pt_add(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "pt_pt_add with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_pt_pt_add(test_context, test_dtype, frac_bits)

    def _test_pt_tf_add(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context

        # This test performs one addition.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a)

        sc = sa + b
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = b + sa
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sd))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            saa = shell_tensor.to_shell_tensor(context, a + max_val)
            se = saa - b
            self.assertAllClose(a + max_val - b, shell_tensor.from_shell_tensor(se))

            bb = b + max_val
            sf = bb - sa
            self.assertAllClose(bb - a, shell_tensor.from_shell_tensor(sf))
        else:
            se = sa - b
            self.assertAllClose(a - b, shell_tensor.from_shell_tensor(se))

            sf = b - sa
            self.assertAllClose(b - a, shell_tensor.from_shell_tensor(sf))

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))

    def test_pt_tf_add(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "pt_tf_add with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_pt_tf_add(test_context, test_dtype, frac_bits)


if __name__ == "__main__":
    tf.test.main()
