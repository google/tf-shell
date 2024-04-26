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

    def _test_expand_dims(self, test_context, dim):
        try:
            # This test performs zero additions.
            a = test_utils.uniform_for_n_adds(test_context, 0)
        except Exception as e:
            print(
                f"Note: Skipping test expand_dims with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        expanded_sa = tf_shell.expand_dims(sa, dim)
        self.assertAllClose(tf.expand_dims(a, dim), tf_shell.to_tensorflow(expanded_sa))

        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)
        expanded_ea = tf_shell.expand_dims(ea, dim)
        self.assertAllClose(
            tf.expand_dims(a, dim),
            tf_shell.to_tensorflow(expanded_ea, test_context.key),
        )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_expand_dims(self):
        for test_context in self.test_contexts:
            for dim in range(1, len(test_context.outer_shape) + 1, 1):
                with self.subTest(
                    f"expand_dims on dimension {dim} with context `{test_context}`."
                ):
                    self._test_expand_dims(test_context, dim)
            for dim in range(-len(test_context.outer_shape) + 1, -1, 1):
                with self.subTest(
                    f"expand_dims on dimension {dim} with context `{test_context}`."
                ):
                    self._test_expand_dims(test_context, dim)

    def _test_expand_dims_and_mul(self, test_context, dim):
        try:
            import copy

            reduced_shape = copy.deepcopy(test_context.outer_shape)
            reduced_shape.insert(0, test_context.shell_context.num_slots)
            del reduced_shape[dim]

            # This test performs 1 multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1, shape=reduced_shape)
            b = test_utils.uniform_for_n_muls(test_context, 1)  # full shape
        except Exception as e:
            print(
                f"Note: Skipping test expand_dims_and_mul with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)
        expanded_ea = tf_shell.expand_dims(ea, dim)
        ec = expanded_ea * b  # expanded_ea should broadcast to match b.shape().

        c = tf.expand_dims(a, dim) * b

        self.assertAllClose(c, tf_shell.to_tensorflow(ec, test_context.key))

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_expand_dims_and_mul(self):
        for test_context in self.test_contexts:
            for dim in range(1, len(test_context.outer_shape) + 1, 1):
                with self.subTest(
                    f"expand_dims and multiply with dim {dim} context `{test_context}`."
                ):
                    self._test_expand_dims_and_mul(test_context, dim)
            for dim in range(-len(test_context.outer_shape) + 1, -1, 1):
                with self.subTest(
                    f"expand_dims and multiply on dimension {dim} with context `{test_context}`."
                ):
                    self._test_expand_dims(test_context, dim)

    def _test_expand_dims_dim_0(self, test_context):
        try:
            # This test performs zero additions.
            a = test_utils.uniform_for_n_adds(test_context, 0)
        except Exception as e:
            print(
                f"Note: Skipping test expand_dims_dim_0 with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        try:
            expanded_sa = tf_shell.expand_dims(sa, 0)
            raise Exception(
                "Should not be able to expand dims at axis 0 of a plaintext."
            )
        except ValueError as e:
            pass

        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)
        try:
            expanded_ea = tf_shell.expand_dims(ea, 0)
            raise Exception(
                "Should not be able to expand dims at axis 0 of an encryption."
            )
        except ValueError as e:
            pass

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_expand_dims_dim_0(self):
        for test_context in self.test_contexts:
            with self.subTest(
                f"expand_dims on first dimension with context `{test_context}`."
            ):
                self._test_expand_dims_dim_0(test_context)

    def _reshape(self, test_context, new_shape):
        try:
            # This test performs zero additions.
            a = test_utils.uniform_for_n_adds(test_context, 0)
        except Exception as e:
            print(
                f"Note: Skipping test reshape with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        expanded_sa = tf_shell.reshape(sa, new_shape)
        self.assertAllClose(
            tf.reshape(a, new_shape), tf_shell.to_tensorflow(expanded_sa)
        )

        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)
        expanded_ea = tf_shell.reshape(ea, new_shape)
        self.assertAllClose(
            tf.reshape(a, new_shape),
            tf_shell.to_tensorflow(expanded_ea, test_context.key),
        )

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_reshape(self):
        for test_context in self.test_contexts:
            for dim in range(1, len(test_context.outer_shape) - 1, 1):
                new_shape = (
                    test_context.outer_shape[:dim]
                    + [
                        test_context.outer_shape[dim]
                        * test_context.outer_shape[dim + 1]
                    ]
                    + test_context.outer_shape[dim + 2 :]
                )
                with self.subTest(
                    f"reshape {test_context.outer_shape} to {new_shape} (dimension {dim}) with context `{test_context}`."
                ):
                    self._reshape(
                        test_context, [test_context.shell_context.num_slots] + new_shape
                    )

    def _test_expand_and_reshape(self, test_context, dim):
        try:
            a = test_utils.uniform_for_n_adds(test_context, 0)  # full shape
        except Exception as e:
            print(
                f"Note: Skipping test expand_dims_and_mul with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)
        expanded_ea = tf_shell.expand_dims(ea, dim)  # Add a dimension.
        reshaped_ea = tf_shell.reshape(
            expanded_ea, a.shape
        )  # Remove the added dimension.

        self.assertAllClose(a, tf_shell.to_tensorflow(reshaped_ea, test_context.key))

    def test_expand_and_reshape(self):
        for test_context in self.test_contexts:
            for dim in range(1, len(test_context.outer_shape) + 1, 1):
                with self.subTest(
                    f"expand_dims and reshape with dim {dim} context `{test_context}`."
                ):
                    self._test_expand_and_reshape(test_context, dim)


if __name__ == "__main__":
    tf.test.main()
