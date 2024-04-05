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
        # Create test contexts used for all multiplication tests.
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

        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    # Num plaintext bits: 22, noise bits: 36
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=1,
                    mul_depth_supported=1,
                )
            )

        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 22, noise bits: 36
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                    mul_depth_supported=1,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)
        eb = tf_shell.to_encrypted(sb, test_context.key)

        ec = ea * eb
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ec, test_context.key))

        # Make sure the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, tf_shell.to_tensorflow(eb, test_context.key))

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)

        ec = ea * sb
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ec, test_context.key))

        ed = sb * ea
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)

        ec = ea * b
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ec, test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)

        ec = ea * b
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ec, test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, tf_shell.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)

        sc = sa * sb
        self.assertAllClose(a * b, tf_shell.to_tensorflow(sc))

        # Check the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))

    def test_pt_pt_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_pt_mul with context `{test_context}`."):
                self._test_pt_pt_mul(test_context)


if __name__ == "__main__":
    tf.test.main()
