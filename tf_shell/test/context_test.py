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


class TestShellContext(tf.test.TestCase):
    def _test_mod_reduce_context(self, eager_mode):
        tf.config.run_functions_eagerly(eager_mode)

        @tf.function
        def test_fn():
            # Num plaintext bits: 48, noise bits: 65
            # Max plaintext value: 127, est error: 3.840%
            context = tf_shell.create_context64(
                log_n=11,
                main_moduli=[288230376151748609, 18014398509506561, 1073153, 1032193],
                plaintext_modulus=281474976768001,
                scaling_factor=1052673,
            )
            key = tf_shell.create_key64(context)

            a = tf.ones([2**11, 2, 3], dtype=tf.float32) * 10
            sa = tf_shell.to_shell_plaintext(a, context)
            ea = tf_shell.to_encrypted(sa, key)

            # Mod reducing should not affect the plaintext value.
            smaller_sa = tf_shell.mod_reduce_tensor64(sa)
            self.assertAllClose(a, tf_shell.to_tensorflow(smaller_sa))

            smaller_ea = tf_shell.mod_reduce_tensor64(ea)
            self.assertAllClose(a, tf_shell.to_tensorflow(smaller_ea, key))

            # Check the arguments were not modified
            self.assertAllClose(a, tf_shell.to_tensorflow(sa))
            self.assertAllClose(a, tf_shell.to_tensorflow(ea, key))

        test_fn()

    def test_mod_reduce_context(self):
        for eager_mode in [False, True]:
            with self.subTest(f"{self._testMethodName} with eager_mode={eager_mode}."):
                self._test_mod_reduce_context(eager_mode)


if __name__ == "__main__":
    tf.test.main()
