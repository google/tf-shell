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
import tf_shell.python.shell_ops as shell_ops
import tensorflow as tf
import typing


class ShellContext64(tf.experimental.ExtensionType):
    _raw_context: tf.Tensor
    is_autocontext: bool
    log_n: tf.Tensor
    num_slots: tf.Tensor
    two_n: tf.Tensor
    main_moduli: tf.Tensor
    level: tf.Tensor
    aux_moduli: tf.Tensor
    plaintext_modulus: tf.Tensor
    noise_variance: int
    scaling_factor: int
    seed: str

    def __init__(
        self,
        _raw_context,
        is_autocontext,
        log_n,
        main_moduli,
        aux_moduli,
        plaintext_modulus,
        noise_variance,
        scaling_factor,
        seed,
    ):
        self._raw_context = _raw_context
        self.is_autocontext = is_autocontext
        self.log_n = tf.convert_to_tensor(log_n, dtype=tf.uint64)
        self.num_slots = 2 ** tf.cast(log_n, dtype=tf.int64)
        self.two_n = self.num_slots * 2
        if isinstance(main_moduli, list):
            main_moduli = tf.convert_to_tensor(main_moduli, dtype=tf.uint64)
        self.main_moduli = main_moduli
        self.level = tf.shape(main_moduli)[0]
        if isinstance(aux_moduli, list):
            aux_moduli = tf.convert_to_tensor(aux_moduli, dtype=tf.uint64)
        self.aux_moduli = aux_moduli
        self.plaintext_modulus = tf.convert_to_tensor(
            plaintext_modulus, dtype=tf.uint64
        )
        self.noise_variance = noise_variance
        self.scaling_factor = scaling_factor
        self.seed = seed

    def _get_generic_context_spec(self):
        return ShellContext64.Spec(
            _raw_context=tf.TensorSpec([], dtype=tf.variant),
            is_autocontext=self.is_autocontext,
            log_n=tf.TensorSpec([], dtype=tf.uint64),
            num_slots=tf.TensorSpec([], dtype=tf.int64),
            two_n=tf.TensorSpec([], dtype=tf.int64),
            main_moduli=tf.TensorSpec(None, dtype=tf.uint64),
            level=tf.TensorSpec([], dtype=tf.int32),
            aux_moduli=tf.TensorSpec(None, dtype=tf.uint64),
            plaintext_modulus=tf.TensorSpec(None, dtype=tf.uint64),
            noise_variance=self.noise_variance,
            scaling_factor=self.scaling_factor,
            seed=self.seed,
        )


def mod_reduce_context64(context):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    smaller_context = shell_ops.modulus_reduce_context64(context._raw_context)

    mod_reduced = ShellContext64(
        _raw_context=smaller_context,
        is_autocontext=context.is_autocontext,
        log_n=context.log_n,
        main_moduli=context.main_moduli[:-1],
        aux_moduli=context.aux_moduli,
        plaintext_modulus=context.plaintext_modulus,
        noise_variance=context.noise_variance,
        scaling_factor=context.scaling_factor,
        seed=context.seed,
    )

    return mod_reduced


def create_context64(
    log_n,
    main_moduli,
    plaintext_modulus,
    aux_moduli=[],
    noise_variance=8,
    scaling_factor=1,
    seed="",
):
    if len(seed) > 64:
        raise ValueError("Seed must be at most 64 characters long.")
    elif len(seed) < 64 and seed != "":
        seed = seed.ljust(64)

    shell_context, _, _, _, _ = shell_ops.context_import64(
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        seed=seed,
    )

    return ShellContext64(
        _raw_context=shell_context,
        is_autocontext=False,
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        scaling_factor=scaling_factor,
        seed=seed,
    )


def create_autocontext64(
    log2_cleartext_sz,
    scaling_factor,
    noise_offset_log2,
    noise_variance=8,
    seed="",
):
    if len(seed) > 64:
        raise ValueError("Seed must be at most 64 characters long.")
    seed = seed.ljust(64)

    shell_context, new_log_n, new_qs, new_ps, new_t = shell_ops.auto_shell_context64(
        log2_cleartext_sz=log2_cleartext_sz,
        scaling_factor=scaling_factor,
        log2_noise_offset=noise_offset_log2,
        noise_variance=noise_variance,
    )

    return ShellContext64(
        _raw_context=shell_context,
        is_autocontext=True,
        log_n=new_log_n,
        main_moduli=new_qs,
        aux_moduli=new_ps,
        plaintext_modulus=new_t,
        noise_variance=noise_variance,
        scaling_factor=scaling_factor,
        seed=seed,
    )
