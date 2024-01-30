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
    rotation_dtypes = [
        tf.int32,
        tf.int64,
        tf.float32,
        tf.float64,
    ]
    roll_test_outer_shape = [3, 3]
    test_outer_shape = [2, 5, 4]

    def _test_keygen(self, test_context):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)
        rotation_key = shell_tensor.create_rotation_key64(context, key)
        assert rotation_key is not None

    def test_keygen(self):
        for test_context in test_utils.test_contexts:
            with self.subTest("keygen"):
                self._test_keygen(test_context)

    # TensorFlow's roll has slightly different sematics than encrypted roll.
    # Encrypted rotation affects top and bottom halves independently.
    # This function emulates this in plaintext by splitting the tensor in half,
    # rotating each half, and then concatenating them back together.
    def plaintext_roll(self, t, shift):
        top, bottom = tf.split(t, num_or_size_splits=2, axis=0)
        top = tf.roll(top, shift, axis=0)
        bottom = tf.roll(bottom, shift, axis=0)
        rotated_tftensor = tf.concat([top, bottom], axis=0)
        return rotated_tftensor

    def _test_roll(self, test_context, key, rotation_key, plaintext_dtype, roll_num):
        context = test_context.shell_context

        # Create a tensor with the shape of slots x (outer_shape) where each
        # column of the first dimensions counts from 0 to slots-1.
        tftensor = tf.range(0, test_context.slots, delta=1, dtype=plaintext_dtype)
        for i in range(len(self.roll_test_outer_shape)):
            tftensor = tf.expand_dims(tftensor, axis=-1)
            tftensor = tf.tile(
                tftensor, multiples=[1] * (i + 1) + [self.roll_test_outer_shape[i]]
            )

        rolled_tftensor = self.plaintext_roll(tftensor, roll_num)

        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)

        rolled_enc = enc.roll(rotation_key, roll_num)
        rolled_result = rolled_enc.get_decrypted(key)
        self.assertAllClose(rolled_tftensor, rolled_result)

    def test_roll(self):
        for test_context in test_utils.test_contexts:
            context = test_context.shell_context
            key = shell_tensor.create_key64(context)
            rotation_key = shell_tensor.create_rotation_key64(context, key)
            rotation_range = test_context.slots // 2 - 1

            for test_dtype in self.rotation_dtypes:
                for roll_num in range(-rotation_range, rotation_range, 1):
                    with self.subTest(
                        "rotate with dtype %s, rotating by %s" % (test_dtype, roll_num)
                    ):
                        self._test_roll(
                            test_context, key, rotation_key, test_dtype, roll_num
                        )

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

    def _test_reduce_sum_axis_0(
        self, test_context, key, rotation_key, plaintext_dtype, frac_bits
    ):
        context = test_context.shell_context

        # reduce_sum across axis 0 requires adding over all the slots.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype,
            test_context.plaintext_modulus,
            frac_bits,
            test_context.slots,
        )

        if max_val is 0:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test reduce_sum_axis0 with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        test_shape = self.test_outer_shape.copy()
        test_shape.insert(0, test_context.slots)

        tftensor = tf.random.uniform(
            test_shape,
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        tftensor = tf.cast(tftensor, plaintext_dtype)
        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)

        enc_reduce_sum = enc.reduce_sum(axis=0, rotation_key=rotation_key)

        tftensor_out = enc_reduce_sum.get_decrypted(key)
        self.assertAllClose(tftensor_out, self.plaintext_reduce_sum_axis_0(tftensor))

    def test_reduce_sum_axis_0(self):
        for test_context in test_utils.test_contexts:
            context = test_context.shell_context
            key = shell_tensor.create_key64(context)
            rotation_key = shell_tensor.create_rotation_key64(context, key)

            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in self.rotation_dtypes:
                    with self.subTest(
                        "reduce_sum_axis_0 with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_reduce_sum_axis_0(
                            test_context, key, rotation_key, test_dtype, frac_bits
                        )

    def _test_reduce_sum_axis_n(
        self, test_context, plaintext_dtype, frac_bits, outer_axis
    ):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)

        # reduce_sum across `axis` requires adding over that dimension.
        min_val, max_val = test_utils.get_bounds_for_n_adds(
            plaintext_dtype,
            test_context.plaintext_modulus,
            frac_bits,
            self.test_outer_shape[outer_axis],
        )

        if max_val == 0:
            # Test parameters do not support reduce_sum at this precision.
            print(
                "Note: Skipping test reduce_sum_axis0 with dtype %s and frac_bits %d. Not enough precision to support this test."
                % (plaintext_dtype, frac_bits)
            )
            return

        test_shape = self.test_outer_shape.copy()
        test_shape.insert(0, test_context.slots)

        tftensor = tf.random.uniform(
            test_shape,
            dtype=tf.int64,
            maxval=max_val,
            minval=min_val,
        )
        tftensor = tf.cast(tftensor, plaintext_dtype)
        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)

        enc_reduce_sum = enc.reduce_sum(axis=outer_axis + 1)

        tftensor_out = enc_reduce_sum.get_decrypted(key)
        self.assertAllClose(tftensor_out, tf.reduce_sum(tftensor, axis=outer_axis + 1))

    def test_reduce_sum_axis_n(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in self.rotation_dtypes:
                    for outer_axis in range(len(self.test_outer_shape)):
                        with self.subTest(
                            "reduce_sum_axis_n with fractional bits %d, dtype %s, and axis %d"
                            % (frac_bits, test_dtype, outer_axis + 1)
                        ):
                            self._test_reduce_sum_axis_n(
                                test_context,
                                test_dtype,
                                frac_bits,
                                outer_axis,
                            )


if __name__ == "__main__":
    tf.test.main()
