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


class TestShellTensorRotation(tf.test.TestCase):
    # plaintext_dtype = tf.int32
    # log_slots = 11
    # slots = 2**log_slots

    # def get_context(self):
    #     return shell_tensor.create_context64(
    #         log_n=self.log_slots,
    #         main_moduli=[8556589057, 8388812801],
    #         aux_moduli=[34359709697],
    #         plaintext_modulus=40961,
    #         noise_variance=8,
    #         seed="",
    #     )

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

    def _test_rotate(self, test_context, plaintext_dtype):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)
        rotation_key = shell_tensor.create_rotation_key64(context, key)

        shift_right = 1
        shift_left = -1

        tftensor = tf.range(test_context.slots, delta=1, dtype=None, name="range")

        tftensor_right = self.plaintext_roll(tftensor, shift_right)
        tftensor_left = self.plaintext_roll(tftensor, shift_left)

        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)

        enc_right = enc.roll(rotation_key, shift_right)
        tftensor_out = enc_right.get_decrypted(key)
        self.assertAllClose(tftensor_out, tftensor_right)

        enc_left = enc.roll(rotation_key, shift_left)
        tftensor_out = enc_left.get_decrypted(key)
        self.assertAllClose(tftensor_out, tftensor_left)

    def test_rotate(self):
        for test_context in test_utils.test_contexts:
            for test_dtype in test_utils.test_dtypes:
                with self.subTest("rotate with dtype %s" % (test_dtype)):
                    self._test_rotate(test_context, test_dtype)

    # TensorFlow's reduce_sum has slightly different semantics than encrypted
    # reduce_sum. Encrypted reduce_sum affects top and bottom halves
    # independently, as well as repeating the sum across the halves. This
    # function emulates this in plaintext.
    def plaintext_reduce_sum(self, t):
        half_slots = t.shape[0] // 2
        bottom_answer = tf.math.reduce_sum(
            t[0 : half_slots], axis=0, keepdims=True
        )
        top_answer = tf.math.reduce_sum(t[half_slots :], axis=0, keepdims=True)

        repeated_bottom_answer = tf.repeat(
            bottom_answer, repeats=half_slots, axis=0
        )
        repeated_top_answer = tf.repeat(top_answer, repeats=half_slots, axis=0)

        return tf.concat([repeated_bottom_answer, repeated_top_answer], 0)

    def _test_reduce_sum(self, test_context, plaintext_dtype, frac_bits):
        context = test_context.shell_context
        key = shell_tensor.create_key64(context)
        rotation_key = shell_tensor.create_rotation_key64(context, key)
        test_shape = [test_context.slots, 2, 3, 4]

        tftensor = tf.random.uniform(test_shape, dtype=tf.int32, maxval=10)
        s = shell_tensor.to_shell_tensor(context, tftensor)
        enc = s.get_encrypted(key)

        enc_reduce_sum = enc.reduce_sum(rotation_key)

        tftensor_out = enc_reduce_sum.get_decrypted(key)
        self.assertAllClose(tftensor_out, self.plaintext_reduce_sum(tftensor))

    def test_reduce_sum(self):
        for test_context in test_utils.test_contexts:
            for frac_bits in test_utils.test_fxp_fractional_bits:
                for test_dtype in test_utils.test_dtypes:
                    with self.subTest(
                        "reduce_sum with fractional bits %d and dtype %s"
                        % (frac_bits, test_dtype)
                    ):
                        self._test_reduce_sum(test_context, test_dtype, frac_bits)


if __name__ == "__main__":
    tf.test.main()
