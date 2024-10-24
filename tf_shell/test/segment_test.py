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
                    generate_rotation_keys=True,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.rotation_test_contexts = None

    # tf-shell segment functions differs from tensorflow in the following ways:
    # First, the ciphertext dimension is included in the output, but only one
    # dimension is valid. For the top half of the ciphertext, the first
    # dimension is valid, and for the bottom half, the `num_slots // 2`th
    # dimension is valid.
    # Second, the reduction only happens across half of the batching dimension,
    # due to how rotations in tf-shell work. Segment reduction happens on the
    # top and bottom halves of the ciphertext independently.
    def plaintext_segment_sum(self, x, segments, num_segments, start_segment=0):
        half_slots = x.shape[0] // 2
        padding = tf.zeros_like(x[:half_slots])

        x_top = tf.concat([x[:half_slots], padding], 0)
        x_bottom = tf.concat([padding, x[half_slots:]], 0)

        top_answer = tf.math.unsorted_segment_sum(x_top, segments, num_segments)
        bottom_answer = tf.math.unsorted_segment_sum(x_bottom, segments, num_segments)

        if start_segment > 0:
            top_answer = tf.concat(
                [
                    tf.zeros_like(top_answer[:start_segment]),
                    top_answer[start_segment:],
                ],
                axis=0,
            )
            bottom_answer = tf.concat(
                [
                    tf.zeros_like(bottom_answer[:start_segment]),
                    bottom_answer[start_segment:],
                ],
                axis=0,
            )

        return top_answer, bottom_answer

    def create_rand_data(self, test_context, repeats):
        try:
            shape_prod = math.prod(test_context.outer_shape)
            num_adds = repeats / 2 * shape_prod
            a = test_utils.uniform_for_n_adds(test_context, num_adds)
            return a
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return None

    def create_uniform_segments(self, test_context, repeats, num_segments):
        segments = tf.range(0, limit=num_segments, dtype=tf.int32)
        segments = tf.random.shuffle(segments)
        for l in range(len(test_context.outer_shape)):
            segments = tf.expand_dims(segments, axis=-1)

        segments = tf.tile(segments, [repeats] + test_context.outer_shape)
        return segments

    def create_nonuniform_segments(self, test_context, repeats, num_segments):
        segments = self.create_uniform_segments(test_context, repeats, num_segments)

        # Create a random mask to set some segments to -1.
        mask = tf.random.uniform(segments.shape, maxval=2, dtype=segments.dtype)
        masked_segments = tf.where(mask > 0, segments, -1)
        return masked_segments

    # def get_inferred_shape(self, ea, segments, num_segments, rot_key):
    #     @tf.function
    #     def shape_inf_func(ea, segments, num_segments, rot_key):
    #         print(f"YOYOYOYOYOYOYOYOYO in {ea}, {segments}, {num_segments}")
    #         ess, counts = tf_shell.segment_sum(ea, segments, num_segments, rot_key)
    #         print(f"YOYOYOYOYOYOYOYOYO {ess}, {counts}")
    #         return ess.shape, counts.shape

    #     return shape_inf_func(ea, segments, num_segments, rot_key)

    def _test_segment_sum(self, test_context, segment_creator_functor):
        repeats = 8
        num_segments = test_context.shell_context.num_slots.numpy() // repeats

        a = self.create_rand_data(test_context, repeats)
        if a is None:
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)

        segments = segment_creator_functor(test_context, repeats, num_segments)
        segments_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            2,
            num_segments,
        ]
        counts_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            num_segments,
        ]

        @tf.function
        def test_functor(ea, segments, num_segments, rot_key):
            ess, counts = tf_shell.segment_sum(ea, segments, num_segments, rot_key)
            # Tests shape inference
            self.assertEqual(ess.shape.ndims, len(segments_shape_should_be))
            for i in range(ess.shape.ndims):
                if ess.shape[i] is not None:
                    self.assertEqual(ess.shape[i], segments_shape_should_be[i])
            self.assertEqual(counts.shape.ndims, len(counts_shape_should_be))
            for i in range(counts.shape.ndims):
                if counts.shape[i] is not None:
                    self.assertEqual(counts.shape[i], counts_shape_should_be[i])

            return ess, counts

        ess, counts = test_functor(
            ea, segments, num_segments, test_context.rotation_key
        )

        ss = tf_shell.to_tensorflow(ess, test_context.key)

        pt_ss_top, pt_ss_bottom = self.plaintext_segment_sum(a, segments, num_segments)

        # Ensure the reduced data is correct.
        self.assertAllClose(pt_ss_top, ss[0][0])
        self.assertAllClose(
            pt_ss_bottom, ss[test_context.shell_context.num_slots // 2][1]
        )

        # Ensure the counts are correct.
        def bincount(x):
            return tf.math.bincount(x, minlength=num_segments, maxlength=num_segments)

        segments_nonnegative = tf.where(segments >= 0, segments, num_segments + 1)
        pt_counts = tf.map_fn(bincount, segments_nonnegative)
        self.assertAllEqual(pt_counts, counts)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_segment_sum(self):
        for test_context in self.test_contexts:
            for segment_creator in [
                self.create_uniform_segments,
                self.create_nonuniform_segments,
            ]:
                with self.subTest(
                    f"{self._testMethodName} with context `{test_context}` and segment creator `{segment_creator}`."
                ):
                    self._test_segment_sum(test_context, segment_creator)

    def _test_segment_sum_no_reduction(self, test_context, segment_creator_functor):
        repeats = 8
        num_segments = test_context.shell_context.num_slots.numpy() // repeats

        a = self.create_rand_data(test_context, repeats)
        if a is None:
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)

        segments = segment_creator_functor(test_context, repeats, num_segments)
        segments_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            num_segments,
        ]
        counts_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            num_segments,
        ]

        @tf.function
        def test_functor(ea, segments, num_segments):
            ess, counts = tf_shell.segment_sum(
                ea, segments, num_segments, reduction="none"
            )
            # Tests shape inference
            self.assertEqual(ess.shape.ndims, len(segments_shape_should_be))
            for i in range(ess.shape.ndims):
                if ess.shape[i] is not None:
                    self.assertEqual(ess.shape[i], segments_shape_should_be[i])
            self.assertEqual(counts.shape.ndims, len(counts_shape_should_be))
            for i in range(counts.shape.ndims):
                if counts.shape[i] is not None:
                    self.assertEqual(counts.shape[i], counts_shape_should_be[i])

            return ess, counts

        ess, counts = test_functor(ea, segments, num_segments)

        ss = tf_shell.to_tensorflow(ess, test_context.key)

        pt_result = tf.math.unsorted_segment_sum(a, segments, num_segments)

        # Ensure the data is correct.
        self.assertAllClose(pt_result, tf.reduce_sum(ss, axis=0))

        # Ensure the counts are correct.
        def bincount(x):
            return tf.math.bincount(x, minlength=num_segments, maxlength=num_segments)

        segments_nonnegative = tf.where(segments >= 0, segments, num_segments + 1)
        pt_counts = tf.map_fn(bincount, segments_nonnegative)
        self.assertAllEqual(pt_counts, counts)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_segment_sum_no_reduction(self):
        for test_context in self.test_contexts:
            for segment_creator in [
                self.create_uniform_segments,
                self.create_nonuniform_segments,
            ]:
                with self.subTest(
                    f"{self._testMethodName} with context `{test_context}` and segment creator `{segment_creator}``."
                ):
                    self._test_segment_sum_no_reduction(test_context, segment_creator)

    def _test_segment_sum_fewer_dims(self, test_context, segment_creator_functor):
        repeats = 8
        num_segments = test_context.shell_context.num_slots.numpy() // repeats

        a = self.create_rand_data(test_context, repeats)
        if a is None:
            return

        sa = tf_shell.to_shell_plaintext(a, test_context.shell_context)
        ea = tf_shell.to_encrypted(sa, test_context.key, test_context.shell_context)

        segments = segment_creator_functor(test_context, repeats, num_segments)

        # Remove the last d-1 dimensions of segments to test fewer_dimsing.
        ndims = len(segments.shape)
        for _ in range(ndims - 2):
            segments = segments[..., 0]

        segments_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            2,
            num_segments,
        ] + test_context.outer_shape[1:]
        counts_shape_should_be = [
            test_context.shell_context.num_slots.numpy(),
            num_segments,
        ]
        # TODO: Should counts shape really be this?
        # counts_shape_should_be = [test_context.shell_context.num_slots.numpy(), num_segments] + test_context.outer_shape[1:]

        @tf.function
        def test_functor(ea, segments, num_segments, rot_key):
            ess, counts = tf_shell.segment_sum(ea, segments, num_segments, rot_key)
            # Tests shape inference
            self.assertEqual(ess.shape.ndims, len(segments_shape_should_be))
            for i in range(ess.shape.ndims):
                if ess.shape[i] is not None:
                    self.assertEqual(ess.shape[i], segments_shape_should_be[i])
            self.assertEqual(counts.shape.ndims, len(counts_shape_should_be))
            for i in range(counts.shape.ndims):
                if counts.shape[i] is not None:
                    self.assertEqual(counts.shape[i], counts_shape_should_be[i])

            return ess, counts

        ess, counts = test_functor(
            ea, segments, num_segments, test_context.rotation_key
        )

        ss = tf_shell.to_tensorflow(ess, test_context.key)

        pt_ss_top, pt_ss_bottom = self.plaintext_segment_sum(a, segments, num_segments)

        # Ensure the data is correctly reduced.
        self.assertAllClose(pt_ss_top, ss[0][0])
        self.assertAllClose(
            pt_ss_bottom, ss[test_context.shell_context.num_slots // 2][1]
        )

        # Ensure the counts are correct.
        def bincount(x):
            return tf.math.bincount(x, minlength=num_segments, maxlength=num_segments)

        segments_nonnegative = tf.where(segments >= 0, segments, num_segments + 1)
        pt_counts = tf.map_fn(bincount, segments_nonnegative)
        self.assertAllEqual(pt_counts, counts)

        # Ensure initial arguments are not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_segment_sum_fewer_dims(self):
        for test_context in self.test_contexts:
            if len(test_context.outer_shape) > 1:
                for segment_creator in [
                    self.create_uniform_segments,
                    self.create_nonuniform_segments,
                ]:
                    with self.subTest(
                        f"{self._testMethodName} with context `{test_context}` and segment creator `{segment_creator}`."
                    ):
                        self._test_segment_sum_fewer_dims(test_context, segment_creator)
            else:
                print(
                    f"Note: Skipping test {self._testMethodName} because outer shape {test_context.outer_shape} is too small."
                )


if __name__ == "__main__":
    tf.test.main()
