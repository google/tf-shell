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
    matmul_fxp_fractional_bits = [0, 1]
    matmul_max_val = 3
    matmul_val_offset = -1
    matmul_dtypes = [
        tf.int32,
        tf.int64,
        tf.float32,
        tf.float64,
    ]

    def _test_ct_ct_mul(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one multiplication.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
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

        sa = shell_tensor.to_shell_tensor(context, a, fxp_fractional_bits=frac_bits)
        sb = shell_tensor.to_shell_tensor(context, b, fxp_fractional_bits=frac_bits)
        ea = sa.get_encrypted(key)
        eb = sb.get_encrypted(key)

        ec = ea * eb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(b, eb.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

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
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one multiplication.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
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
        sa = shell_tensor.to_shell_tensor(context, a, fxp_fractional_bits=frac_bits)
        sb = shell_tensor.to_shell_tensor(context, b, fxp_fractional_bits=frac_bits)
        ea = sa.get_encrypted(key)

        ec = ea * sb
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

        ed = sb * ea
        self.assertAllClose(a * b, ed.get_decrypted(key))

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
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one multiplication.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
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
            [],
            dtype=tf.int32,
            maxval=max_val,
            minval=min_val,
        )
        b = tf.cast(b, plaintext_dtype)
        sa = shell_tensor.to_shell_tensor(context, a, fxp_fractional_bits=frac_bits)
        ea = sa.get_encrypted(key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(key))

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
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # This test performs one multiplication.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
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
        sa = shell_tensor.to_shell_tensor(context, a, fxp_fractional_bits=frac_bits)
        ea = sa.get_encrypted(key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(key))
        self.assertAllClose(a * b, ec.get_decrypted(key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(key))

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
        context = test_context.shell_context

        # This test performs one multiplication.
        min_val, max_val = test_utils.get_bounds_for_n_muls(
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
        sa = shell_tensor.to_shell_tensor(context, a, fxp_fractional_bits=frac_bits)
        sb = shell_tensor.to_shell_tensor(context, b, fxp_fractional_bits=frac_bits)

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
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        a = (
            tf.random.uniform(
                [test_context.slots, 5], dtype=tf.int32, maxval=self.matmul_max_val
            )
            - self.matmul_val_offset
        )
        a = tf.cast(a, plaintext_dtype)
        b = (
            tf.random.uniform([5, 7], dtype=tf.int32, maxval=self.matmul_max_val)
            - self.matmul_val_offset
        )
        b = tf.cast(b, plaintext_dtype)

        ea = shell_tensor.to_shell_tensor(
            context, a, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)

        ec = shell_tensor.matmul(ea, b)
        self.assertAllClose(a, ea.get_decrypted(key))
        c = tf.matmul(a, b)
        self.assertAllClose(c, ec.get_decrypted(key))

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
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)
        rotation_key = shell_tensor.create_rotation_key64(context, key)

        a = (
            tf.random.uniform(
                [3, 5, test_context.slots],
                dtype=tf.int32,
                maxval=self.matmul_max_val,
            )
            - self.matmul_val_offset
        )
        a = tf.cast(a, plaintext_dtype)
        b = (
            tf.random.uniform(
                [test_context.slots, 2], dtype=tf.int32, maxval=self.matmul_max_val
            )
            - self.matmul_val_offset
        )
        b = tf.cast(b, plaintext_dtype)

        eb = shell_tensor.to_shell_tensor(
            context, b, fxp_fractional_bits=frac_bits
        ).get_encrypted(key)

        ec = shell_tensor.matmul(a, eb, rotation_key)
        self.assertAllClose(b, eb.get_decrypted(key))

        check_c = self.plaintext_matmul(a, b)
        self.assertAllClose(check_c.shape, ec.shape)
        self.assertAllClose(check_c, ec.get_decrypted(key))

    def test_tf_ct_matmul(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in self.matmul_dtypes:
                    with self.subTest(
                        "tf_ct_matmul with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_tf_ct_matmul(test_context, test_dtype, frac_bits)


if __name__ == "__main__":
    tf.test.main()
