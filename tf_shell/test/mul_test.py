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
import test_utils


class TestShellTensor(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        int_dtypes = [
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.uint32,
            tf.int32,
            tf.uint64,
            tf.int64,
        ]
        cls.test_contexts = []

        # Create test contexts for all integer datatypes. While this test
        # performs at most two multiplications, since the scaling factor is 1
        # there are no restrictions on the main moduli.
        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    log_n=11,
                    main_moduli=[288230376151748609, 288230376151760897],
                    aux_moduli=[],
                    plaintext_modulus=65537,
                    noise_variance=8,
                    scaling_factor=1,
                    mul_depth_supported=2,
                    seed="",
                )
            )

        # Create test contexts for floating point datatypes at various scaling
        # factors. Since these test perform at two multiplications, the trailing
        # two main moduli should be roughly equal to the scaling factor, for
        # modulus reduction during each multiplication.
        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    log_n=11,
                    main_moduli=[
                        288230376151748609,
                        288230376151760897,
                        147457,
                        114689,
                    ],
                    aux_moduli=[],
                    plaintext_modulus=1099511795713,
                    noise_variance=8,
                    scaling_factor=131073,
                    mul_depth_supported=2,
                    seed="",
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    # # Matrix multiplication tests require special parameters to avoid overflow.
    # matmul_test_contexts = []

    # # Create test contexts for all integer datatypes.
    # for int_dtype in test_utils.int_dtypes:
    #     matmul_test_contexts.append(
    #         test_utils.TestContext(
    #             outer_shape=[],  # dummy
    #             log_slots=11,
    #             main_moduli=[288230376151748609, 18014398509506561, 1073153, 1032193],
    #             plaintext_modulus=281474976768001,
    #             plaintext_dtype=int_dtype,
    #             scaling_factor=1,
    #             mul_depth_supported=2,
    #         )
    #     )

    # # Create test contexts for floating point datatypes at various scaling factors.
    # for float_dtype in [tf.float32, tf.float64]:
    #     matmul_test_contexts.append(
    #         test_utils.TestContext(
    #             outer_shape=[],  # dummy
    #             log_slots=11,
    #             main_moduli=[
    #                 288230376151748609,
    #                 18014398509506561,
    #                 1073153,
    #                 1032193,
    #             ],
    #             plaintext_modulus=281474976768001,
    #             plaintext_dtype=float_dtype,
    #             scaling_factor=1052673,
    #             mul_depth_supported=0,
    #         )
    #     )

    def _test_ct_ct_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_ct_mul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_tensor(test_context.shell_context, a)
        sb = tf_shell.to_shell_tensor(test_context.shell_context, b)
        ea = sa.get_encrypted(test_context.key)
        eb = sb.get_encrypted(test_context.key)

        ec = ea * eb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(b, eb.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

    def test_ct_ct_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_ct_mul with context `{test_context}`."):
                self._test_ct_ct_mul(test_context)

    def _test_ct_pt_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_pt_mul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_tensor(test_context.shell_context, a)
        sb = tf_shell.to_shell_tensor(test_context.shell_context, b)
        ea = sa.get_encrypted(test_context.key)

        ec = ea * sb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = sb * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_pt_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_pt_mul with context `{test_context}`."):
                self._test_ct_pt_mul(test_context)

    def _test_ct_tf_scalar_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_scalar_mul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_tensor(test_context.shell_context, a)
        ea = sa.get_encrypted(test_context.key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_tf_scalar_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_tf_scalar_mul with context `{test_context}`."):
                self._test_ct_tf_scalar_mul(test_context)

    def _test_ct_tf_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_pt_mul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_tensor(test_context.shell_context, a)
        ea = sa.get_encrypted(test_context.key)

        ec = ea * b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a * b, ec.get_decrypted(test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, ed.get_decrypted(test_context.key))

    def test_ct_tf_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_tf_mul with context `{test_context}`."):
                self._test_ct_tf_mul(test_context)

    def _test_pt_pt_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test pt_pt_mul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_tensor(test_context.shell_context, a)
        sb = tf_shell.to_shell_tensor(test_context.shell_context, b)

        sc = sa * sb
        self.assertAllClose(a * b, tf_shell.from_shell_tensor(sc))

    def test_pt_pt_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_pt_mul with context `{test_context}`."):
                self._test_pt_pt_mul(test_context)

    def _test_ct_tf_matmul(self, test_context):
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 5],
                subsequent_adds=5,  # For dim(1)
            )
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=[5, 7])
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_matmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_shell_tensor(test_context.shell_context, a).get_encrypted(
            test_context.key
        )

        ec = tf_shell.matmul(ea, b)
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        c = tf.matmul(a, b)
        self.assertAllClose(
            c,
            ec.get_decrypted(test_context.key),
            # , atol=5 * 2 ** (-frac_bits - 1)
        )

    def test_ct_tf_matmul(self):
        for test_context in self.test_contexts:
            if test_context.plaintext_dtype not in [
                tf.int32,
                tf.int64,
                tf.float32,
                tf.float64,
            ]:
                print(
                    f"TensorFlow does not support matmul with dtype {test_context.plaintext_dtype}. Skipping test."
                )
                continue
            with self.subTest(f"ct_tf_matmul with context `{test_context}`."):
                self._test_ct_tf_matmul(test_context)

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

    def _test_tf_ct_matmul(self, test_context):
        # Generating the following tensors should always succeed since this test
        # uses it's own special context.
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[3, 5, test_context.shell_context.num_slots],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
            b = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 2],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
        except Exception as e:
            print(
                f"Note: Skipping test tf_ct_matmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        eb = tf_shell.to_shell_tensor(test_context.shell_context, b).get_encrypted(
            test_context.key
        )

        ec = tf_shell.matmul(a, eb, test_context.rotation_key)
        self.assertAllClose(
            b,
            eb.get_decrypted(test_context.key),
        )

        check_c = self.plaintext_matmul(a, b)
        self.assertAllClose(
            check_c,
            ec.get_decrypted(test_context.key),
            # atol=test_context.shell_context.num_slots * 2 ** (-frac_bits - 1),
        )

    def test_tf_ct_matmul(self):
        for test_context in self.test_contexts:
            if test_context.plaintext_dtype not in [
                tf.int32,
                tf.int64,
                tf.float32,
                tf.float64,
            ]:
                print(
                    f"TensorFlow does not support matmul with dtype {test_context.plaintext_dtype}. Skipping test."
                )
                continue
            with self.subTest(f"tf_ct_matmul with context `{test_context}`."):
                self._test_tf_ct_matmul(test_context)


if __name__ == "__main__":
    tf.test.main()
