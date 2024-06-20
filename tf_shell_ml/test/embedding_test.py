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
import unittest
import tensorflow as tf
import keras
import numpy as np
import tf_shell
import tf_shell_ml

# # Num plaintext bits: 32, noise bits: 84
# # Max representable value: 654624
# context = tf_shell.create_context64(
#     log_n=11,
#     main_moduli=[288230376151748609, 144115188076060673],
#     plaintext_modulus=4294991873,
#     scaling_factor=3,
#     mul_depth_supported=3,
#     seed="test_seed",
# )
# 61 bits of security according to lattice estimator primal_bdd.
# Runtime 170 seconds (83ms/example).

# Num plaintext bits: 32, noise bits: 84
# Max representable value: 654624
context = tf_shell.create_context64(
    log_n=12,
    main_moduli=[288230376151760897, 288230376152137729],
    plaintext_modulus=4294991873,
    scaling_factor=3,
    mul_depth_supported=3,
    seed="test_seed",
)
# 120 bits of security according to lattice estimator primal_bdd.
# Runtime 388 seconds (95ms/example).

key = tf_shell.create_key64(context)
rotation_key = tf_shell.create_rotation_key64(context, key)


class TestEmbedding(tf.test.TestCase):
    def test_embedding_forward(self):
        input_dim = 100
        output_dim = 10
        embedding_layer = tf_shell_ml.ShellEmbedding(input_dim, output_dim)

        # First check plaintext forward pass.
        x = tf.zeros((context.num_slots, 5), dtype=tf.int64)
        y = embedding_layer(x)

        # Check that the output is the same for the same inputs.
        for i in range(1, context.num_slots):
            self.assertAllEqual(y[0], y[i])

        # Next check encrypted forward pass throws an error.
        enc_x = tf_shell.to_encrypted(x, key, context)
        try:
            should_fail = embedding_layer(enc_x)
        except:
            pass
        else:
            raise ValueError(
                "Embedding layer forward with encrypted value should fail."
            )

    def _test_embedding(self):
        input_dim = 100
        output_dim = 10
        embedding_layer = tf_shell_ml.ShellEmbedding(input_dim, output_dim)

        special_index = 2
        x = tf.ones((context.num_slots, 1), dtype=tf.int64) * special_index

        @tf.function
        def forward_backward(x):
            y = embedding_layer(x)

            dy = tf.ones_like(y)
            enc_dy = tf_shell.to_encrypted(dy, key, context)

            embedding_layer.backward_accum(enc_dy, rotation_key)
            dx = embedding_layer.decrypt_grad(key)
            return dx

        dx = forward_backward(x)

        self.assertAllEqual(
            dx[special_index, :], tf.constant(context.num_slots, shape=(output_dim,))
        )

        # Make sure the rest of the gradient elements are 0
        for i in range(0, input_dim):
            if i == special_index:
                self.assertAllEqual(
                    dx[special_index, :],
                    tf.constant(context.num_slots, shape=(output_dim,)),
                )
            else:
                self.assertAllEqual(dx[i, :], tf.constant(0, shape=(output_dim,)))

    def test_embedding_eager(self):
        tf.config.run_functions_eagerly(True)
        self._test_embedding()

    def test_embedding_defer(self):
        tf.config.run_functions_eagerly(False)
        self._test_embedding()


if __name__ == "__main__":
    unittest.main()
