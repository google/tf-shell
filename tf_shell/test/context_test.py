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
    def test_create_context(self):
        # Num plaintext bits: 48, noise bits: 65
        # Max plaintext value: 1, est error: 0.003%
        context = tf_shell.create_context64(
            log_n=11,
            main_moduli=[288230376151748609, 18014398509506561, 8404993, 8380417],
            aux_moduli=[],
            plaintext_modulus=281474976768001,
            scaling_factor=8392705,
            mul_depth_supported=2,
        )

        ql = context.main_moduli[-1]
        ql2 = context.main_moduli[-2]

        # The ratio between the smaller and the larger context should be
        # the last modulus in the chain.
        smaller_context = context.get_mod_reduced()
        self.assertAllClose(context.Q / smaller_context.Q, ql)

        # The ratio between the smaller and the larger context should be the
        # scaling factor.
        even_smaller_context = smaller_context.get_mod_reduced()
        self.assertAllClose(context.Q / even_smaller_context.Q, ql * ql2)

    def test_mod_reduce_context(self):
        # Num plaintext bits: 48, noise bits: 65
        # Max plaintext value: 127, est error: 3.840%
        context = tf_shell.create_context64(
            log_n=11,
            main_moduli=[288230376151748609, 18014398509506561, 1073153, 1032193],
            aux_moduli=[],
            plaintext_modulus=281474976768001,
            scaling_factor=1052673,
            mul_depth_supported=2,
        )
        key = tf_shell.create_key64(context)

        a = tf.ones([2**11, 2, 3], dtype=tf.float32) * 10
        sa = tf_shell.to_shell_plaintext(a, context)
        ea = tf_shell.to_encrypted(sa, key)

        # Mod reducing should not affect the plaintext value.
        smaller_sa = sa.get_mod_reduced()
        self.assertAllClose(a, tf_shell.to_tensorflow(smaller_sa))

        smaller_ea = ea.get_mod_reduced()
        self.assertAllClose(a, tf_shell.to_tensorflow(smaller_ea, key))

        # Check the arguments were not modified
        self.assertAllClose(a, tf_shell.to_tensorflow(sa))
        self.assertAllClose(a, tf_shell.to_tensorflow(ea, key))


if __name__ == "__main__":
    tf.test.main()
