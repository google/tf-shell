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
        # This test performs zero additions, just negation.
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 0)

        if a is None:
            # Test parameters do not support zero additions at this
            # precision.
            print(
                "Note: Skipping test neg with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a, frac_bits)
        nsa = -sa
        self.assertAllClose(-a, shell_tensor.from_shell_tensor(nsa))

        ea = sa.get_encrypted(test_context.key)
        nea = -ea
        self.assertAllClose(-a, nea.get_decrypted(test_context.key))

        self.assertAllClose(a, ea.get_decrypted(test_context.key))

    def test_neg(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in [1]:
                # for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in [tf.float32]:
                    # for test_dtype in test_utils.test_dtypes:
                    if test_dtype.is_unsigned:
                        # Negating an unsigned value is undefined.
                        continue
                    with self.subTest(
                        "neg with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_neg(test_context, test_dtype, frac_bits)

    def _test_ct_ct_add(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one addition.
        _, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support one addition at this
            # precision.
            print(
                "Note: Skipping test ct_ct_add with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        ea = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, frac_bits
        ).get_encrypted(test_context.key)
        eb = shell_tensor.to_shell_tensor(
            test_context.shell_context, b, frac_bits
        ).get_encrypted(test_context.key)

        ec = ea + eb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(b, eb.get_decrypted(test_context.key))
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val, frac_bits
            ).get_encrypted(test_context.key)
            ed = eaa - eb
            self.assertAllClose(a + max_val - b, ed.get_decrypted(test_context.key))
        else:
            ed = ea - eb
            self.assertAllClose(a - b, ed.get_decrypted(test_context.key))

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
        # This test performs one addition.
        _, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support one addition at this
            # precision.
            print(
                "Note: Skipping test ct_pt_add with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a, frac_bits)
        sb = shell_tensor.to_shell_tensor(test_context.shell_context, b, frac_bits)
        ea = sa.get_encrypted(test_context.key)

        ec = ea + sb
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key))

        ed = sb + ea
        self.assertAllClose(a + b, ed.get_decrypted(test_context.key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val, frac_bits
            ).get_encrypted(test_context.key)
            ee = eaa - sb
            self.assertAllClose(a + max_val - b, ee.get_decrypted(test_context.key))

            sbb = shell_tensor.to_shell_tensor(
                test_context.shell_context, b + max_val, frac_bits
            )
            ef = sbb - ea
            self.assertAllClose(b + max_val - a, ef.get_decrypted(test_context.key))
        else:
            ee = ea - sb
            self.assertAllClose(a - b, ee.get_decrypted(test_context.key))

            ef = sb - ea
            self.assertAllClose(b - a, ef.get_decrypted(test_context.key))

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
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
        # This test performs one addition.
        _, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support one addition at this
            # precision.
            print(
                "Note: Skipping test ct_tf_add with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a, frac_bits)
        ea = sa.get_encrypted(test_context.key)

        ec = ea + b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key))

        ed = b + ea
        self.assertAllClose(a + b, ed.get_decrypted(test_context.key))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val, frac_bits
            ).get_encrypted(test_context.key)
            ee = eaa - b
            self.assertAllClose(a + max_val - b, ee.get_decrypted(test_context.key))

            bb = b + max_val
            ef = bb - ea
            self.assertAllClose(bb - a, ef.get_decrypted(test_context.key))
        else:
            ee = ea - b
            self.assertAllClose(a - b, ee.get_decrypted(test_context.key))

            ef = b - ea
            self.assertAllClose(b - a, ef.get_decrypted(test_context.key))

        # Ensure initial arguemnts are not modified.
        self.assertAllClose(a, ea.get_decrypted(test_context.key))

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
        # This test performs one addition.
        _, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support one addition at this
            # precision.
            print(
                "Note: Skipping test pt_pt_add with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a, frac_bits)
        sb = shell_tensor.to_shell_tensor(test_context.shell_context, b, frac_bits)

        sc = sa + sb
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            saa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val, frac_bits
            )
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
        # This test performs one addition.
        _, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 1
        )
        a = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_adds(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support one addition at this
            # precision.
            print(
                "Note: Skipping test pt_tf_add with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a, frac_bits)

        sc = sa + b
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc))

        sd = b + sa
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sd))

        if plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            saa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val, frac_bits
            )
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
