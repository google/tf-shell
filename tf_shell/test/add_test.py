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
                    scaling_factor=1,
                    mul_depth_supported=0,
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
                        scaling_factor=scaling_factor,
                        mul_depth_supported=0,
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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        nsa = -sa
        self.assertAllClose(-a, tf_shell.to_tensorflow(nsa))

        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)
        nea = -ea
        self.assertAllClose(-a, tf_shell.to_tensorflow(nea, test_context.key))

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

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

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)
        eb = tf_shell.to_encrypted(b, test_context.key, test_context.shell_context)

        ec = ea + eb
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        # Make sure the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, tf_shell.to_tensorflow(eb, test_context.key))

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid overflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ed = eaa - eb
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
            )
        else:
            ed = ea - eb
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
            )

    def test_ct_ct_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_ct_add with context `{test_context}`."):
                self._test_ct_ct_add(test_context)

    def _test_ct_ct_add_with_broadcasting(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test ct_ct_add_with_broadcasting with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)
        eb = tf_shell.to_encrypted(b, test_context.key, test_context.shell_context)

        ec = ea + eb
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        # Make sure the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, tf_shell.to_tensorflow(eb, test_context.key))

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid overflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ed = eaa - eb
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
            )
        else:
            ed = ea - eb
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
            )

    def test_ct_ct_add_with_broadcasting(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"ct_ct_add_with_broadcasting with context `{test_context}`."
            ):
                self._test_ct_ct_add_with_broadcasting(test_context)

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)

        ec = ea + sb
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        ed = sb + ea
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        # Make sure the arguments were not modified.
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ee = eaa - sb
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            sbb = tf_shell.to_shell_plaintext(b + max_val, test_context.shell_context)
            ef = sbb - ea
            self.assertAllClose(
                b + max_val - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )
        else:
            ee = ea - sb
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            ef = sb - ea
            self.assertAllClose(
                b - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))

    def test_ct_pt_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_pt_add with context `{test_context}`."):
                self._test_ct_pt_add(test_context)

    def _test_ct_pt_add_with_broadcasting(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test ct_pt_add_with_broadcasting with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key)

        ec = ea + sb
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        ed = sb + ea
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        # Make sure the arguments were not modified.
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ee = eaa - sb
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            sbb = tf_shell.to_shell_plaintext(b + max_val, test_context.shell_context)
            ef = sbb - ea
            self.assertAllClose(
                b + max_val - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )
        else:
            ee = ea - sb
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            ef = sb - ea
            self.assertAllClose(
                b - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))

    def test_ct_pt_add_with_broadcasting(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"ct_pt_add_with_broadcasting with context `{test_context}`."
            ):
                self._test_ct_pt_add_with_broadcasting(test_context)

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

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)

        ec = ea + b
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        ed = b + ea
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ee = eaa - b
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            bb = b + max_val
            ef = bb - ea
            self.assertAllClose(
                bb - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )
        else:
            ee = ea - b
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            ef = b - ea
            self.assertAllClose(
                b - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_ct_tf_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_tf_add with context `{test_context}`."):
                self._test_ct_tf_add(test_context)

    def _test_ct_tf_add_with_broadcasting(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_add_with_broadcasting with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)

        ec = ea + b
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        ed = b + ea
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ee = eaa - b
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            bb = b + max_val
            ef = bb - ea
            self.assertAllClose(
                bb - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )
        else:
            ee = ea - b
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            ef = b - ea
            self.assertAllClose(
                b - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_ct_tf_add_with_broadcasting(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"ct_tf_add_with_broadcasting with context `{test_context}`."
            ):
                self._test_ct_tf_add_with_broadcasting(test_context)

    def _test_ct_list_add_with_padding(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_add_with_broadcasting with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)

        # Convert b to a python list where the first dimension is only 2.
        # When encoded, this will be padded to match the first dimension of a,
        # i.e. the number of slots in the ciphertext.
        b_list = b[:2, ...].numpy().tolist()

        # Set the unused elements of b to 0 for checking purposes.
        b = tf.concat([b[:2, ...], tf.zeros_like(b[2:, ...])], axis=0)

        ec = ea + b_list
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ec, test_context.key), atol=1e-3
        )

        ed = b_list + ea
        self.assertAllClose(
            a + b, tf_shell.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            eaa = tf_shell.to_encrypted(
                a + max_val, test_context.key, test_context.shell_context
            )
            ee = eaa - b_list
            self.assertAllClose(
                a + max_val - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            bb = b + max_val
            bb_list = bb[:2, ...].numpy().tolist()
            ef = bb_list - ea
            self.assertAllClose(
                (bb - a)[:2, ...],
                tf_shell.to_tensorflow(ef, test_context.key)[:2, ...],
                atol=1e-3,
            )
        else:
            ee = ea - b_list
            self.assertAllClose(
                a - b, tf_shell.to_tensorflow(ee, test_context.key), atol=1e-3
            )

            ef = b_list - ea
            self.assertAllClose(
                b - a, tf_shell.to_tensorflow(ef, test_context.key), atol=1e-3
            )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_ct_list_add_with_padding(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"ct_list_add_with_padding with context `{test_context}`."
            ):
                self._test_ct_list_add_with_padding(test_context)

    def _test_ct_scalar_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test ct_scalar_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        # Resize b so the size of the first dimension is 1. This is the
        # ciphertext packing dimension.
        b = tf.expand_dims(b[0], axis=0)

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)

        sc = sa + b
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            ee = saa - b
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(ee), atol=1e-3)
        else:
            sd = sa - b
            self.assertAllClose(a - b, tf_shell.to_tensorflow(sd), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))

    def test_ct_scalar_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_scalar_add with context `{test_context}`."):
                self._test_ct_scalar_add(test_context)

    def _test_ct_single_scalar_add(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=[1])
        except Exception as e:
            print(
                f"Note: Skipping test ct_scalar_add with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)

        sc = sa + b
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            ee = saa - b
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(ee), atol=1e-3)
        else:
            sd = sa - b
            self.assertAllClose(a - b, tf_shell.to_tensorflow(sd), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))

    def test_ct_single_scalar_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_scalar_add with context `{test_context}`."):
                self._test_ct_scalar_add(test_context)

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)

        sc = sa + sb
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            ee = saa - sb
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(ee), atol=1e-3)
        else:
            sd = sa - sb
            self.assertAllClose(a - b, tf_shell.to_tensorflow(sd), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))

    def test_pt_pt_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_pt_add with context `{test_context}`."):
                self._test_pt_pt_add(test_context)

    def _test_pt_pt_add_with_broadcast(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test pt_pt_add_with_broadcast with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        sb = tf_shell.to_shell_plaintext(b, test_context.shell_context)

        sc = sa + sb
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            ee = saa - sb
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(ee), atol=1e-3)
        else:
            sd = sa - sb
            self.assertAllClose(a - b, tf_shell.to_tensorflow(sd), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(b, tf_shell.to_tensorflow(sb))

    def test_pt_pt_add_with_broadcast(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"pt_pt_add_with_broadcast with context `{test_context}`."
            ):
                self._test_pt_pt_add_with_broadcast(test_context)

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

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)

        sc = sa + b
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        sd = b + sa
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sd), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            se = saa - b
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(se), atol=1e-3)

            bb = b + max_val
            sf = bb - sa
            self.assertAllClose(bb - a, tf_shell.to_tensorflow(sf), atol=1e-3)
        else:
            se = sa - b
            self.assertAllClose(a - b, tf_shell.to_tensorflow(se), atol=1e-3)

            sf = b - sa
            self.assertAllClose(b - a, tf_shell.to_tensorflow(sf), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))

    def test_pt_tf_add(self):
        for test_context in self.test_contexts:
            with self.subTest(f"pt_tf_add with context `{test_context}`."):
                self._test_pt_tf_add(test_context)

    def _test_pt_tf_add_with_broadcast(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test pt_tf_add_with_broadcast with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)

        sc = sa + b
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        sd = b + sa
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sd), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            se = saa - b
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(se), atol=1e-3)

            bb = b + max_val
            sf = bb - sa
            self.assertAllClose(bb - a, tf_shell.to_tensorflow(sf), atol=1e-3)
        else:
            se = sa - b
            self.assertAllClose(a - b, tf_shell.to_tensorflow(se), atol=1e-3)

            sf = b - sa
            self.assertAllClose(b - a, tf_shell.to_tensorflow(sf), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))

    def test_pt_tf_add_with_broadcast(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"pt_tf_add_with_broadcast with context `{test_context}`."
            ):
                self._test_pt_tf_add_with_broadcast(test_context)

    def _test_pt_list_add_with_padding(self, test_context):
        try:
            # This test performs one addition.
            _, max_val = test_utils.get_bounds_for_n_adds(test_context, 1)
            a = test_utils.uniform_for_n_adds(test_context, 1)
            b = test_utils.uniform_for_n_adds(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test pt_tf_add_with_broadcast with context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        # Convert b to a python list where the first dimension is only 2.
        # When encoded, this will be padded to match the first dimension of a,
        # i.e. the number of slots in the ciphertext.
        b_list = b[:2, ...].numpy().tolist()

        # Set the unused elements of b to 0 for checking purposes.
        b = tf.concat([b[:2, ...], tf.zeros_like(b[2:, ...])], axis=0)

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)

        sc = sa + b_list
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sc), atol=1e-3)

        sd = b_list + sa
        self.assertAllClose(a + b, tf_shell.to_tensorflow(sd), atol=1e-3)

        if test_context.plaintext_dtype.is_unsigned:
            # To test subtraction, ensure that a > b to avoid underflow.
            # a + max_val is safe, because max_val is the total range / 2 and
            # a is less than max_val.
            max_val = int(max_val)
            saa = tf_shell.to_shell_plaintext(a + max_val, test_context.shell_context)
            se = saa - b_list
            self.assertAllClose(a + max_val - b, tf_shell.to_tensorflow(se), atol=1e-3)

            bb = b + max_val
            bb_list = bb[:2, ...].numpy().tolist()
            sf = bb_list - sa
            self.assertAllClose((bb - a)[:2], tf_shell.to_tensorflow(sf)[:2], atol=1e-3)
        else:
            se = sa - b
            self.assertAllClose(a - b, tf_shell.to_tensorflow(se), atol=1e-3)

            sf = b - sa
            self.assertAllClose(b - a, tf_shell.to_tensorflow(sf), atol=1e-3)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))

    def test_pt_list_add_with_padding(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"pt_list_add_with_padding with context `{test_context}`."
            ):
                self._test_pt_list_add_with_padding(test_context)


if __name__ == "__main__":
    tf.test.main()
