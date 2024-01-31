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
    # Matrix multiplication tests require smaller parameters to avoid overflow.
    matmul_dtypes = [
        tf.int32,
        tf.int64,
        tf.float32,
        tf.float64,
    ]
    matmul_contexts = [
        # Num plaintext bits: 27, noise bits: 66, num rns moduli: 2
        test_utils.TestContext(
            outer_shape=[],  # dummy
            log_slots=11,
            main_moduli=[281474976768001, 281474976829441],
            plaintext_modulus=134246401,
        ),
    ]

    def _test_ct_ct_mul(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one multiplication.
        a = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test ct_ct_mul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        )
        sb = shell_tensor.to_shell_tensor(
            test_context.shell_context, b, fxp_fractional_bits=frac_bits
        )
        ea = sa.get_encrypted(test_context.key)
        eb = sb.get_encrypted(test_context.key)

        ec = ea * eb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(b, eb.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

    def test_ct_ct_mul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_ct_mul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_ct_mul(test_context, test_dtype, frac_bits)

    def _test_ct_pt_mul(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one multiplication.
        a = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test ct_pt_mul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        )
        sb = shell_tensor.to_shell_tensor(
            test_context.shell_context, b, fxp_fractional_bits=frac_bits
        )
        ea = sa.get_encrypted(test_context.key)

        ec = ea * sb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = sb * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_pt_mul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_pt_mul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_pt_mul(test_context, test_dtype, frac_bits)

    def _test_ct_tf_scalar_mul(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one multiplication.
        a = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test ct_tf_scalar_mul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        )
        ea = sa.get_encrypted(test_context.key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_tf_scalar_mul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_tf_scalar_mul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_tf_scalar_mul(test_context, test_dtype, frac_bits)

    def _test_ct_tf_mul(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one multiplication.
        a = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test ct_pt_mul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        )
        ea = sa.get_encrypted(test_context.key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_tf_mul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "ct_tf_mul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_tf_mul(test_context, test_dtype, frac_bits)

    def _test_pt_pt_mul(self, test_context, plaintext_dtype, frac_bits):
        # This test performs one multiplication.
        a = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)
        b = test_utils.uniform_for_n_muls(plaintext_dtype, test_context, frac_bits, 1)

        if a is None:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test pt_pt_mul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        sa = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        )
        sb = shell_tensor.to_shell_tensor(
            test_context.shell_context, b, fxp_fractional_bits=frac_bits
        )

        sc = sa * sb
        self.assertAllClose(a * b, shell_tensor.from_shell_tensor(sc))

    def test_pt_pt_mul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "pt_pt_mul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_pt_pt_mul(test_context, test_dtype, frac_bits)

    def _test_ct_tf_matmul(self, test_context, plaintext_dtype, frac_bits):
        a = test_utils.uniform_for_n_muls(
            plaintext_dtype,
            test_context,
            frac_bits,
            1,
            shape=[test_context.slots, 5],
            subsequent_adds=5,  # For dim(1)
        )
        b = test_utils.uniform_for_n_muls(
            plaintext_dtype, test_context, frac_bits, 1, shape=[5, 7]
        )

        if a is None or b is None:
            print(
                "Note: Skipping test ct_tf_matmul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        ea = shell_tensor.to_shell_tensor(
            test_context.shell_context, a, fxp_fractional_bits=frac_bits
        ).get_encrypted(test_context.key)

        ec = shell_tensor.matmul(ea, b)
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        c = tf.matmul(a, b)
        self.assertAllClose(
            c,
            ec.get_decrypted(test_context.key)
            # , atol=5 * 2 ** (-frac_bits - 1)
        )

    def test_ct_tf_matmul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in self.matmul_dtypes:
                    with self.subTest(
                        "ct_tf_matmul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_ct_tf_matmul(test_context, test_dtype, frac_bits)

    # tf-shell matmult has slightly different semantics than plaintext /
    # Tensorflow. Encrypted matmult affects top and bottom halves independently,
    # as well as the first dimension repeating the sum of either the halves.
    # This function emulates this in plaintext.
    def plaintext_matmul(self, a, b):
        half_slots = b.shape[0] // 2
        tf_half_slots = tf.constant([half_slots], dtype=tf.int32)

        a_shape = tf.shape(a)
        a_top_start = tf.zeros_like(a_shape)
        a_top_shape = tf.concat([a_shape[:-1], tf_half_slots], axis=0)
        a_top = tf.slice(a, a_top_start, a_top_shape)
        a_bottom_start = tf.concat(
            [tf.zeros_like(a_top_start[:-1]), tf_half_slots], axis=0
        )
        a_bottom_shape = tf.concat([a_shape[:-1], tf_half_slots], axis=0)
        a_bottom = tf.slice(a, a_bottom_start, a_bottom_shape)

        assert len(tf.shape(b)) == 2
        b_top = b[:half_slots, :]
        b_bottom = b[half_slots:, :]

        top = tf.matmul(a_top, b_top)
        bottom = tf.matmul(a_bottom, b_bottom)

        top = tf.expand_dims(top, axis=0)
        bottom = tf.expand_dims(bottom, axis=0)

        top = tf.repeat(top, repeats=[half_slots], axis=0)
        bottom = tf.repeat(bottom, repeats=[half_slots], axis=0)

        return tf.concat([top, bottom], axis=0)

    def _test_tf_ct_matmul(self, test_context, plaintext_dtype, frac_bits):
        a = test_utils.uniform_for_n_muls(
            plaintext_dtype,
            test_context,
            frac_bits,
            1,
            shape=[3, 5, test_context.slots],
            subsequent_adds=test_context.slots / 2,
        )
        b = test_utils.uniform_for_n_muls(
            plaintext_dtype,
            test_context,
            frac_bits,
            1,
            shape=[test_context.slots, 2],
            subsequent_adds=test_context.slots / 2,
        )
        if a is None or b is None:
            print(
                "Note: Skipping test tf_ct_matmul with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        eb = shell_tensor.to_shell_tensor(
            test_context.shell_context, b, fxp_fractional_bits=frac_bits
        ).get_encrypted(test_context.key)

        ec = shell_tensor.matmul(a, eb, test_context.rotation_key)
        self.assertAllClose(
            b,
            eb.get_decrypted(test_context.key),
        )

        check_c = self.plaintext_matmul(a, b)
        self.assertAllClose(
            check_c,
            ec.get_decrypted(test_context.key),
            # atol=test_context.slots * 2 ** (-frac_bits - 1),
        )

    def test_tf_ct_matmul(self):
        for test_context in self.matmul_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in self.matmul_dtypes:
                    with self.subTest(
                        "tf_ct_matmul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_tf_ct_matmul(test_context, test_dtype, frac_bits)


if __name__ == "__main__":
    tf.test.main()
