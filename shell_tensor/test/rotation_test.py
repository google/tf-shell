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
from multiprocessing import Pool
from itertools import repeat


class TestShellTensorRotation(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        cls.test_contexts = []

        # TensorFlow only supports rotation for int32 and int64 integer types.
        # Create test contexts for all integer datatypes. While this test
        # performs at most two multiplications, since the scaling factor is 1
        # there are no restrictions on the main moduli.
        for int_dtype in [tf.int32, tf.int64]:
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
                    mul_depth_supported=0,
                    seed="",
                )
            )

        # Create test contexts for floating point datatypes at a real-world
        # scaling factor.
        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 40, noise bits: 70
                    # Max plaintext value: 3640, est error: 0.000%
                    log_n=11,
                    main_moduli=[288230376151748609, 2251799813824513, 12289],
                    aux_moduli=[],
                    plaintext_modulus=1099511795713,
                    noise_variance=8,
                    scaling_factor=12289,
                    mul_depth_supported=1,
                    seed="",
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    # TensorFlow's roll has slightly different semantics than encrypted roll.
    # Encrypted rotation affects top and bottom halves independently.
    # This function emulates this in plaintext by splitting the tensor in half,
    # rotating each half, and then concatenating them back together.
    def plaintext_roll(self, t, shift):
        top, bottom = tf.split(t, num_or_size_splits=2, axis=0)
        top = tf.roll(top, shift, axis=0)
        bottom = tf.roll(bottom, shift, axis=0)
        rotated_tftensor = tf.concat([top, bottom], axis=0)
        return rotated_tftensor

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

        rolled_tftensor = self.plaintext_roll(tftensor, roll_num)

        s = shell_tensor.to_shell_tensor(test_context.shell_context, tftensor)
        enc = s.get_encrypted(test_context.key)

        rolled_enc = enc.roll(test_context.rotation_key, roll_num)
        rolled_result = rolled_enc.get_decrypted(test_context.key)
        self.assertAllClose(rolled_tftensor, rolled_result, atol=1e-3)

    def test_roll(self):
        for test_context in self.test_contexts:
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

        rolled_tftensor = self.plaintext_roll(tftensor, roll_num)

        s = shell_tensor.to_shell_tensor(test_context.shell_context, tftensor)
        enc = s.get_encrypted(test_context.key)

        # Test roll on a mod reduced ciphertext.
        enc_reduced = enc.get_mod_reduced()
        rolled_enc_reduced = enc_reduced.roll(test_context.rotation_key, roll_num)
        rolled_result_reduced = rolled_enc_reduced.get_decrypted(test_context.key)
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

    # TensorFlow's reduce_sum has slightly different semantics than encrypted
    # reduce_sum. Encrypted reduce_sum affects top and bottom halves
    # independently, as well as repeating the sum across the halves. This
    # function emulates this in plaintext.
    def plaintext_reduce_sum_axis_0(self, t):
        half_slots = t.shape[0] // 2
        bottom_answer = tf.math.reduce_sum(t[0:half_slots], axis=0, keepdims=True)
        top_answer = tf.math.reduce_sum(t[half_slots:], axis=0, keepdims=True)

        repeated_bottom_answer = tf.repeat(bottom_answer, repeats=half_slots, axis=0)
        repeated_top_answer = tf.repeat(top_answer, repeats=half_slots, axis=0)

        return tf.concat([repeated_bottom_answer, repeated_top_answer], 0)

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

        s = shell_tensor.to_shell_tensor(test_context.shell_context, tftensor)
        enc = s.get_encrypted(test_context.key)

        enc_reduce_sum = enc.reduce_sum(axis=0, rotation_key=test_context.rotation_key)

        tftensor_out = enc_reduce_sum.get_decrypted(test_context.key)
        self.assertAllClose(
            tftensor_out, self.plaintext_reduce_sum_axis_0(tftensor), atol=1e-3
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

        s = shell_tensor.to_shell_tensor(test_context.shell_context, tftensor)
        enc = s.get_encrypted(test_context.key)

        enc_reduce_sum = enc.reduce_sum(axis=outer_axis + 1)

        tftensor_out = enc_reduce_sum.get_decrypted(test_context.key)
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
