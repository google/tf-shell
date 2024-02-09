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

        # Create test contexts for all integer datatypes.
        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    log_n=11,
                    main_moduli=[8556589057, 8388812801],
                    aux_moduli=[],
                    plaintext_modulus=40961,
                    noise_variance=8,
                    scaling_factor=1,
                    mul_depth_supported=0,
                    seed="",
                )
            )

        # Create test contexts for floating point datatypes at various scaling
        # factors. For addition, the scaling factor does not need to have
        # anything to do with the choice of main_moduli. This is not true for
        # multiplication.
        for float_dtype in [tf.float32, tf.float64]:
            for scaling_factor in [1, 2, 3, 7]:
                cls.test_contexts.append(
                    test_utils.TestContext(
                        outer_shape=[3, 2, 3],
                        plaintext_dtype=float_dtype,
                        log_n=11,
                        main_moduli=[8556589057, 8388812801],
                        aux_moduli=[],
                        plaintext_modulus=40961,
                        noise_variance=8,
                        scaling_factor=scaling_factor,
                        mul_depth_supported=0,
                        seed="",
                    )
                )

    @classmethod
    def tearDownClass(cls):
        cls.rotation_test_contexts = None

    def _test_neg(self, test_context):
        try:
            # This test performs zero additions, just negation.
            a = test_utils.uniform_for_n_adds(test_context, 0)
        except Exception as e:
            print(
                f"Note: Skipping test neg with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a)
        nsa = -sa
        self.assertAllClose(-a, shell_tensor.from_shell_tensor(nsa))

        ea = sa.get_encrypted(test_context.key)
        nea = -ea
        self.assertAllClose(-a, nea.get_decrypted(test_context.key))

        self.assertAllClose(a, ea.get_decrypted(test_context.key))

    def test_neg(self):
        for test_context in self.test_contexts:
            if test_context.plaintext_dtype.is_unsigned:
                # Negating an unsigned value is undefined.
                continue
            with self.subTest(f"neg with context `{test_context}`."):
                self._test_neg(test_context)

    def _test_ct_ct_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_ct_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = shell_tensor.to_shell_tensor(test_context.shell_context, a).get_encrypted(
            test_context.key
        )
        eb = shell_tensor.to_shell_tensor(test_context.shell_context, b).get_encrypted(
            test_context.key
        )

        ec = ea + eb
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(b, eb.get_decrypted(test_context.key))
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid overflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val
            ).get_encrypted(test_context.key)
            ed = eaa - eb
            self.assertAllClose(
                a + max_val - b, ed.get_decrypted(test_context.key), atol=1e-3
            )
        else:
            ed = ea - eb
            self.assertAllClose(a - b, ed.get_decrypted(test_context.key), atol=1e-3)

    def test_ct_ct_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_ct_add with context `{test_context}`."):
                self._test_ct_ct_add(test_context)

    def _test_ct_pt_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_pt_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a)
        sb = shell_tensor.to_shell_tensor(test_context.shell_context, b)
        ea = sa.get_encrypted(test_context.key)

        ec = ea + sb
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key), atol=1e-3)

        ed = sb + ea
        self.assertAllClose(a + b, ed.get_decrypted(test_context.key), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val
            ).get_encrypted(test_context.key)
            ee = eaa - sb
            self.assertAllClose(
                a + max_val - b, ee.get_decrypted(test_context.key), atol=1e-3
            )

            sbb = shell_tensor.to_shell_tensor(test_context.shell_context, b + max_val)
            ef = sbb - ea
            self.assertAllClose(
                b + max_val - a, ef.get_decrypted(test_context.key), atol=1e-3
            )
        else:
            ee = ea - sb
            self.assertAllClose(a - b, ee.get_decrypted(test_context.key), atol=1e-3)

            ef = sb - ea
            self.assertAllClose(b - a, ef.get_decrypted(test_context.key), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_ct_pt_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_pt_add with context `{test_context}`."):
                self._test_ct_pt_add(test_context)

    def _test_ct_tf_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a)
        ea = sa.get_encrypted(test_context.key)

        ec = ea + b
        self.assertAllClose(a, ea.get_decrypted(test_context.key))
        self.assertAllClose(a + b, ec.get_decrypted(test_context.key), atol=1e-3)

        ed = b + ea
        self.assertAllClose(a + b, ed.get_decrypted(test_context.key), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = shell_tensor.to_shell_tensor(
                test_context.shell_context, a + max_val
            ).get_encrypted(test_context.key)
            ee = eaa - b
            self.assertAllClose(
                a + max_val - b, ee.get_decrypted(test_context.key), atol=1e-3
            )

            bb = b + max_val
            ef = bb - ea
            self.assertAllClose(bb - a, ef.get_decrypted(test_context.key), atol=1e-3)
        else:
            ee = ea - b
            self.assertAllClose(a - b, ee.get_decrypted(test_context.key), atol=1e-3)

            ef = b - ea
            self.assertAllClose(b - a, ef.get_decrypted(test_context.key), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, ea.get_decrypted(test_context.key))

    def test_ct_tf_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_tf_add with context `{test_context}`."):
                self._test_ct_tf_add(test_context)

    def _test_pt_pt_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test pt_pt_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a)
        sb = shell_tensor.to_shell_tensor(test_context.shell_context, b)

        sc = sa + sb
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = shell_tensor.to_shell_tensor(test_context.shell_context, a + max_val)
            ee = saa - sb
            self.assertAllClose(
                a + max_val - b, shell_tensor.from_shell_tensor(ee), atol=1e-3
            )
        else:
            sd = sa - sb
            self.assertAllClose(a - b, shell_tensor.from_shell_tensor(sd), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))
        self.assertAllClose(b, shell_tensor.from_shell_tensor(sb))

    def test_pt_pt_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_pt_add with context `{test_context}`."):
                self._test_pt_pt_add(test_context)

    def _test_pt_tf_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test pt_tf_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = shell_tensor.to_shell_tensor(test_context.shell_context, a)

        sc = sa + b
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sc), atol=1e-3)

        sd = b + sa
        self.assertAllClose(a + b, shell_tensor.from_shell_tensor(sd), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = shell_tensor.to_shell_tensor(test_context.shell_context, a + max_val)
            se = saa - b
            self.assertAllClose(
                a + max_val - b, shell_tensor.from_shell_tensor(se), atol=1e-3
            )

            bb = b + max_val
            sf = bb - sa
            self.assertAllClose(bb - a, shell_tensor.from_shell_tensor(sf), atol=1e-3)
        else:
            se = sa - b
            self.assertAllClose(a - b, shell_tensor.from_shell_tensor(se), atol=1e-3)

            sf = b - sa
            self.assertAllClose(b - a, shell_tensor.from_shell_tensor(sf), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, shell_tensor.from_shell_tensor(sa))

    def test_pt_tf_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_tf_add with context `{test_context}`."):
                self._test_pt_tf_add(test_context)


if __name__ == "__main__":
    tf.test.main()
