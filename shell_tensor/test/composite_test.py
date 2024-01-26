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
    def _test_ct_ct_mulmul(self, test_context, plaintext_dtype, frac_bits):
        shell_context = test_context.shell_context
        key = shell_tensor.create_key64(shell_context)

        # This test performs two multiplications.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 2
        )
        if max_val == 0:
            # Test parameters do not support two multiplications at this
            # precision.
            print(
                "Note: Skipping test ct_ct_mulmul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)

        ea = shell_tensor.to_shell_tensor(
            shell_context, a, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)
        eb = shell_tensor.to_shell_tensor(
            shell_context, b, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)

        ec = ea * eb
        self.assertAllClose(a * b, ec.get_decrypted(key))

        # Here, ec has a mul_count of 1 while eb has a mul_count of 0. To
        # multiply them, eb needs to be left shifted by the number of fractional
        # bits in the fixed point representation to match ec. ShellTensor should
        # handle this automatically.
        ed = ec * eb
        self.assertAllClose(a * b * b, ed.get_decrypted(key))

        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))

    def test_ct_ct_mulmul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_ct_mulmul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_ct_mulmul(test_context, test_dtype, frac_bits)

    def _test_ct_pt_mulmul(self, test_context, plaintext_dtype, frac_bits):
        shell_context = test_context.shell_context
        key = shell_tensor.create_key64(shell_context)

        # This test performs two multiplications.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
            plaintext_dtype, test_context.plaintext_modulus, frac_bits, 2
        )
        if max_val == 0:
            # Test parameters do not support two multiplications at this
            # precision.
            print(
                "Note: Skipping test ct_pt_mulmul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        a = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        a = tf.cast(a, plaintext_dtype)
        b = tf.random.uniform(
            [test_context.slots, 2, 3, 4],
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)

        ea = shell_tensor.to_shell_tensor(
            shell_context, a, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)
        eb = shell_tensor.to_shell_tensor(
            shell_context, b, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)

        ec = ea * eb
        self.assertAllClose(a * b, ec.get_decrypted(key))

        # Here, ec has a mul_count of 1 while eb has a mul_count of 0. To
        # multiply them, eb needs to be left shifted by the number of fractional
        # bits in the fixed point representation to match ec. ShellTensor should
        # handle this automatically.
        ed = ec * b
        self.assertAllClose(a * b * b, ed.get_decrypted(key))

        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))

    def test_ct_pt_mulmul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_pt_mulmul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_pt_mulmul(test_context, test_dtype, frac_bits)


if __name__ == "__main__":
    tf.test.main()
