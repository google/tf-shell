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
from multiprocessing import Pool
from itertools import repeat


class TestShellTensorRotation(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        # TensorFlow only supports rotation for int32, int64, float32, and float64 types.
        # Create test contexts for all integer datatypes.
        cls.test_contexts = []

        for int_dtype in [tf.int32, tf.int64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 3],
                    plaintext_dtype=int_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=1,
                    mul_depth_supported=1,
                    generate_rotation_keys=True,
                )
            )

        # Create test contexts for floating point datatypes at a real-world
        # scaling factor.
        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033, 12289],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                    mul_depth_supported=1,
                    generate_rotation_keys=True,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_roll(self, test_context, roll_num):
        # Create a tensor with the shape of slots x (outer_shape) where each
        # column of the first dimensions counts from 0 to slots-1. First check
        # if this tensor can actually be represented in the given context. A
        # large scaling factor would be what reduced the maximal representable
        # value.
        _, max_val = test_utils.get_bounds_for_n_adds(test_context, 0)
        if max_val < test_context.shell_context.num_slots:
            print(
                f"Note: Skipping test roll with context {test_context}. Not enough precision to support this test. Context supports max val of {max_val} but need {test_context.shell_context.num_slots}."
            )
            return

        tftensor = tf.range(
            0,
            test_context.shell_context.num_slots,
            delta=1,
            dtype=test_context.plaintext_dtype,
        )
        # Expand dimensions to match the outer shape.
        for i in range(len(test_context.outer_shape)):
            tftensor = tf.expand_dims(tftensor, axis=-1)
            tftensor = tf.tile(
                tftensor, multiples=[1] * (i + 1) + [test_context.outer_shape[i]]
            )

        rolled_tftensor = test_utils.plaintext_roll(tftensor, roll_num)

        s = tf_shell.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = tf_shell.to_encrypted(s, test_context.key)

        rolled_enc = tf_shell.roll(enc, roll_num, test_context.rotation_key)
        rolled_result = tf_shell.to_tensorflow(rolled_enc, test_context.key)
        self.assertAllClose(rolled_tftensor, rolled_result, atol=1e-3)

    def test_roll(self):
        # Testing all contexts for all possible rotations is slow. Instead,
        # test a subset of rotations for each context, and one context tests
        # all rotations.
        for test_context in self.test_contexts:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in [-rotation_range, rotation_range, -1, 0, 1]:
                with self.subTest(
                    f"roll with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll(test_context, roll_num)

        for test_context in [self.test_contexts[0]]:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in range(-rotation_range, rotation_range, 1):
                with self.subTest(
                    f"roll with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll(test_context, roll_num)

    def _test_roll_mod_reduced(self, test_context, roll_num):
        # Create a tensor with the shape of slots x (outer_shape) where each
        # column of the first dimensions counts from 0 to slots-1. First check
        # if this tensor can actually be represented in the given context. A
        # large scaling factor would be what reduced the maximal representable
        # value.
        _, max_val = test_utils.get_bounds_for_n_adds(test_context, 0)
        if max_val < test_context.shell_context.num_slots:
            print(
                f"Note: Skipping test roll with context {test_context}. Not enough precision to support this test. Context supports max val of {max_val} but need {test_context.shell_context.num_slots}."
            )
            return

        tftensor = tf.range(
            0,
            test_context.shell_context.num_slots,
            delta=1,
            dtype=test_context.plaintext_dtype,
        )
        # Expand dimensions to match the outer shape.
        for i in range(len(test_context.outer_shape)):
            tftensor = tf.expand_dims(tftensor, axis=-1)
            tftensor = tf.tile(
                tftensor, multiples=[1] * (i + 1) + [test_context.outer_shape[i]]
            )

        rolled_tftensor = test_utils.plaintext_roll(tftensor, roll_num)

        s = tf_shell.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = tf_shell.to_encrypted(s, test_context.key)

        # Test roll on a mod reduced ciphertext.
        enc_reduced = tf_shell.mod_reduce_tensor64(enc)
        rolled_enc_reduced = tf_shell.roll(
            enc_reduced, roll_num, test_context.rotation_key
        )
        rolled_result_reduced = tf_shell.to_tensorflow(
            rolled_enc_reduced, test_context.key
        )
        self.assertAllClose(rolled_tftensor, rolled_result_reduced, atol=1e-3)

    def test_roll_mod_reduced(self):
        for test_context in self.test_contexts:
            if test_context.shell_context.mul_depth_supported == 0:
                continue
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in range(-rotation_range, rotation_range, 1):
                with self.subTest(
                    f"roll_mod_reduced with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll_mod_reduced(test_context, roll_num)

    def _test_reduce_sum_axis_0(self, test_context):
        # reduce_sum across axis 0 requires adding over all the slots.
        try:
            tftensor = test_utils.uniform_for_n_adds(
                test_context, num_adds=test_context.shell_context.num_slots / 2
            )
        except Exception as e:
            print(
                f"Note: Skipping test reduce_sum_axis_0 with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        s = tf_shell.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = tf_shell.to_encrypted(s, test_context.key)

        enc_reduce_sum = tf_shell.reduce_sum(
            enc, axis=0, rotation_key=test_context.rotation_key
        )

        tftensor_out = tf_shell.to_tensorflow(enc_reduce_sum, test_context.key)
        self.assertAllClose(
            tftensor_out, test_utils.plaintext_reduce_sum_axis_0(tftensor), atol=1e-3
        )

    def test_reduce_sum_axis_0(self):
        for test_context in self.test_contexts:
            with self.subTest(f"reduce_sum_axis_0 with context {test_context}"):
                self._test_reduce_sum_axis_0(test_context)

    def _test_reduce_sum_axis_n(self, test_context, outer_axis):
        # reduce_sum across `axis` requires adding over that dimension.
        try:
            num_adds = test_context.outer_shape[outer_axis]
            tftensor = test_utils.uniform_for_n_adds(test_context, num_adds)
        except Exception as e:
            print(
                f"Note: Skipping test reduce_sum_axis_n with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        s = tf_shell.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = tf_shell.to_encrypted(s, test_context.key)

        enc_reduce_sum = tf_shell.reduce_sum(enc, axis=outer_axis + 1)

        tftensor_out = tf_shell.to_tensorflow(enc_reduce_sum, test_context.key)
        self.assertAllClose(
            tftensor_out, tf.reduce_sum(tftensor, axis=outer_axis + 1), atol=1e-3
        )

    def test_reduce_sum_axis_n(self):
        for test_context in self.test_contexts:
            for outer_axis in range(len(test_context.outer_shape)):
                with self.subTest(
                    f"reduce_sum_axis_n with context {test_context}, and axis {outer_axis+1}"
                ):
                    self._test_reduce_sum_axis_n(test_context, outer_axis)


if __name__ == "__main__":
    tf.test.main()
