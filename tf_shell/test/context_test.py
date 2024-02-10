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
            noise_variance=8,
            scaling_factor=8392705,
            mul_depth_supported=2,
            seed="",
        )

        # The ratio between the smaller and the larger context should be the
        # scaling factor.
        smaller_context = context.get_mod_reduced()
        self.assertAllClose(context.Q / smaller_context.Q, 8392705, rtol=1e-2)

        # The ratio between the smaller and the larger context should be the
        # scaling factor.
        even_smaller_context = smaller_context.get_mod_reduced()
        self.assertAllClose(
            context.Q / even_smaller_context.Q, 8392705 * 8392705, rtol=1e-2
        )

    def test_to_smaller_context(self):
        # Num plaintext bits: 48, noise bits: 65
        # Max plaintext value: 127, est error: 3.840%
        context = tf_shell.create_context64(
            log_n=11,
            main_moduli=[288230376151748609, 18014398509506561, 1073153, 1032193],
            aux_moduli=[],
            plaintext_modulus=281474976768001,
            noise_variance=8,
            scaling_factor=1052673,
            mul_depth_supported=2,
            seed="",
        )
        key = tf_shell.create_key64(context)

        a = tf.ones([2**11], dtype=tf.float32) * 10
        sa = tf_shell.to_shell_tensor(context, a)
        ea = sa.get_encrypted(key)

        # scaled_a = tf.cast(tf.round(a * 8392705 * 8392705), tf.int64)
        # descaled_a = tf.cast(scaled_a, float) / (8392705* 8392705)
        # self.assertAllClose(a, descaled_a)

        # Switching context without any overrides should not affect the
        # plaintext value.
        smaller_sa = sa.get_mod_reduced()
        self.assertAllClose(a, smaller_sa.get_decrypted(key))

        smaller_ea = ea.get_mod_reduced()
        self.assertAllClose(a, smaller_ea.get_decrypted(key))

        # Check the arguments were not modified
        self.assertAllClose(a, sa.get_decrypted())
        self.assertAllClose(a, ea.get_decrypted(key))

    def test_to_smaller_context_with_divide(self):
        # # Num plaintext bits: 48, noise bits: 65
        # # Max plaintext value: 127, est error: 3.840%
        # context = tf_shell.create_context64(
        #     log_n=11,
        #     main_moduli=[288230376151748609, 18014398509506561, 1073153, 1032193],
        #     aux_moduli=[],
        #     plaintext_modulus=281474976768001,
        #     noise_variance=8,
        #     scaling_factor=1052673,
        #     mul_depth_supported=2,
        #     seed="",
        # )
        # Num plaintext bits: 48, noise bits: 30
        # Max plaintext value: 931915, est error: 0.000%
        context = tf_shell.create_context64(
            log_n=11,
            main_moduli=[288230376151748609, 557057, 12289],
            aux_moduli=[],
            plaintext_modulus=281474976768001,
            noise_variance=8,
            scaling_factor=12289,
            mul_depth_supported=1,
            seed="",
        )
        key = tf_shell.create_key64(context)

        a = tf.ones([2**11], dtype=tf.float32) * 931915
        sa = tf_shell.to_shell_tensor(context, a)
        ea = sa.get_encrypted(key)

        # Switch context with preserve_plaintext=False should divide the
        # plaintext value by the scaling factor.
        smaller_sa = sa.get_mod_reduced(preserve_plaintext=False)
        self.assertAllClose(a / context.main_moduli[-1], smaller_sa.get_decrypted(key))

        smaller_ea = ea.get_mod_reduced(preserve_plaintext=False)
        self.assertAllClose(a / context.main_moduli[-1], smaller_ea.get_decrypted(key))

        # Check the arguments were not modified
        self.assertAllClose(a, sa.get_decrypted())
        self.assertAllClose(a, ea.get_decrypted(key))


if __name__ == "__main__":
    tf.test.main()
