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
        # Create test contexts used for matrix multiplication tests. TensorFlow
        # only supports matmul with int32, int64, float32, and float64.
        int_dtypes = [
            tf.int32,
            tf.int64,
        ]
        cls.test_contexts = []

        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
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
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                    mul_depth_supported=1,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_ct_tf_matmul(self, test_context):
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 5],
                subsequent_adds=5,  # For dim(1)
            )
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=[5, 7])
        except Exception as e:
            print(
                f"Note: Skipping test ct_tf_matmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        ea = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)

        ec = tf_shell.matmul(ea, b)
        c = tf.matmul(a, b)
        self.assertAllClose(
            c,
            tf_shell.to_tensorflow(ec, test_context.key),
        )

        # Check the arguments were not modified.
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, test_context.key))

    def test_ct_tf_matmul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_tf_matmul with context `{test_context}`."):
                self._test_ct_tf_matmul(test_context)

    # tf-shell matmult has slightly different semantics than plaintext /
    # Tensorflow. Encrypted matmult affects top and bottom halves independently,
    # as well as the first dimension repeating the sum of either the halves.
    # This function emulates this in plaintext.
    def plaintext_matmul(self, a, b):
        half_slots = b.shape[0] // 2
        tf_half_slots = tf.constant([half_slots], dtype=tf.int32)

        a_shape = tf.shape(a)
        a_top_start = tf.zeros_like(a_shape)
        a_top_shape = tf.concat([a_shape[:-1], tf_half_slots], axis=0)
        a_top = tf.slice(a, a_top_start, a_top_shape)
        a_bottom_start = tf.concat(
            [tf.zeros_like(a_top_start[:-1]), tf_half_slots], axis=0
        )
        a_bottom_shape = tf.concat([a_shape[:-1], tf_half_slots], axis=0)
        a_bottom = tf.slice(a, a_bottom_start, a_bottom_shape)

        assert len(tf.shape(b)) == 2
        b_top = b[:half_slots, :]
        b_bottom = b[half_slots:, :]

        top = tf.matmul(a_top, b_top)
        bottom = tf.matmul(a_bottom, b_bottom)

        top = tf.expand_dims(top, axis=0)
        bottom = tf.expand_dims(bottom, axis=0)

        top = tf.repeat(top, repeats=[half_slots], axis=0)
        bottom = tf.repeat(bottom, repeats=[half_slots], axis=0)

        return tf.concat([top, bottom], axis=0)

    def _test_tf_ct_matmul(self, test_context):
        # Generating the following tensors should always succeed since this test
        # uses it's own special context.
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[3, 5, test_context.shell_context.num_slots],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
            b = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 2],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
        except Exception as e:
            print(
                f"Note: Skipping test tf_ct_matmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        eb = tf_shell.to_encrypted(b, test_context.key, test_context.shell_context)

        ec = tf_shell.matmul(a, eb, test_context.rotation_key)

        check_c = self.plaintext_matmul(a, b)
        self.assertAllClose(
            check_c,
            tf_shell.to_tensorflow(ec, test_context.key),
        )

        # Check the arguments were not modified.
        self.assertAllClose(b, tf_shell.to_tensorflow(eb, test_context.key))

    def test_tf_ct_matmul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"tf_ct_matmul with context `{test_context}`."):
                self._test_tf_ct_matmul(test_context)


if __name__ == "__main__":
    tf.test.main()
