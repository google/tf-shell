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
import math


class TestShellTensor(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        cls.test_contexts = []

        for shape in [[], [1], [2], [2, 1, 3]]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=shape,
                    plaintext_dtype=tf.int32,
                    log_n=11,
                    main_moduli=[8556589057, 8388812801],
                    aux_moduli=[],
                    plaintext_modulus=40961,
                    scaling_factor=1,
                    mul_depth_supported=0,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.rotation_test_contexts = None

    # tf-shell segment functions differs from tensorflow in the following ways:
    # First, the ciphertext dimension is included in the output, but only the
    # dimension is valid.
    # Second, the reduction only happens across half of the batching dimension,
    # due to how rotations in tf-shell work. In other words, it's like segment
    # reduction happens on the top and bottom halves of the ciphertext
    # independently.
    def plaintext_segment_sum(self, x, segments, num_segments):
        half_slots = x.shape[0] // 2
        padding = tf.zeros_like(x[:half_slots])
        x_top = tf.concat([x[:half_slots], padding], 0)
        x_bottom = tf.concat([padding, x[half_slots:]], 0)
        top_answer = tf.math.unsorted_segment_sum(x_top, segments, num_segments)
        bottom_answer = tf.math.unsorted_segment_sum(x_bottom, segments, num_segments)
        # return tf.stack([top_answer, bottom_answer], axis=0)
        return top_answer, bottom_answer

    def get_inferred_shape(self, ea, segments, num_segments, rot_key):
        @tf.function
        def shape_inf_func(ea, segments, num_segments, rot_key):
            ess = tf_shell.segment_sum(ea, segments, num_segments, rot_key)
            return ess.shape

        return shape_inf_func(ea, segments, num_segments, rot_key)

    def _test_segment_sum_same_shape(self, test_context):
        repeats = 8
        num_segments = test_context.shell_context.num_slots // repeats
        try:
            # This test performs `repeats`/2 additions for each dim.
            shape_prod = math.prod(test_context.outer_shape)
            num_adds = repeats / 2 * shape_prod
            a = test_utils.uniform_for_n_adds(test_context, num_adds)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)

        segments = tf.range(num_segments, dtype=tf.int32)
        segments = tf.random.shuffle(segments)
        for l in range(len(test_context.outer_shape)):
            segments = tf.expand_dims(segments, axis=-1)
        segments = tf.tile(segments, [repeats] + test_context.outer_shape)

        ess = tf_shell.segment_sum(
            ea, segments, num_segments, test_context.rotation_key
        )

        # Check shape inference function (used in non-eager mode) matches the
        # real output.
        inf_shape = self.get_inferred_shape(ea, segments, num_segments, test_context.rotation_key)
        self.assertAllClose(ess.shape, inf_shape)

        ss = tf_shell.to_tensorflow(ess, test_context.key)

        pt_ss_top, pt_ss_bottom = self.plaintext_segment_sum(a, segments, num_segments)

        self.assertAllClose(pt_ss_top, ss[0][0])
        self.assertAllClose(pt_ss_bottom, ss[test_context.shell_context.num_slots // 2][1])

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_segment_sum_same_shape(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_segment_sum_same_shape(test_context)

    def _test_segment_sum_broadcast(self, test_context):
        repeats = 8
        num_segments = test_context.shell_context.num_slots // repeats
        try:
            # This test performs `repeats`/2 additions for each dim.
            shape_prod = math.prod(test_context.outer_shape)
            num_adds = repeats / 2 * shape_prod
            a = test_utils.uniform_for_n_adds(test_context, num_adds)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)

        segments = tf.range(num_segments, dtype=tf.int32)
        segments = tf.tile(segments, [repeats])
        # Below makes segments dims match ea, except for the last dim which is
        # broadcast in the segment sum op.
        for i in range(len(test_context.outer_shape) - 1):
            segments = tf.expand_dims(segments, axis=-1)
        segments = tf.tile(segments, [1] + test_context.outer_shape[:-1])

        ess = tf_shell.segment_sum(
            ea, segments, num_segments, test_context.rotation_key
        )

        # Check shape inference function (used in non-eager mode) matches the
        # real output.
        inf_shape = self.get_inferred_shape(ea, segments, num_segments, test_context.rotation_key)
        self.assertAllClose(ess.shape, inf_shape)

        ss = tf_shell.to_tensorflow(ess, test_context.key)

        pt_ss_top, pt_ss_bottom = self.plaintext_segment_sum(a, segments, num_segments)

        self.assertAllClose(pt_ss_top, ss[0][0])
        self.assertAllClose(pt_ss_bottom, ss[test_context.shell_context.num_slots // 2][1])

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_segment_sum_broadcast(self):
        for test_context in self.test_contexts:
            if len(test_context.outer_shape) > 0:
                with self.subTest(
                    f"{self._testMethodName} with context `{test_context}`."
                ):
                    self._test_segment_sum_broadcast(test_context)
            else:
                print(
                    f"Note: Skipping test {self._testMethodName} because outer shape {test_context.outer_shape} is too small."
                )


if __name__ == "__main__":
    tf.test.main()
