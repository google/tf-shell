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
import tf_shell.python.ops.shell_ops as shell_ops
import tensorflow as tf
import typing


class ShellContext64(tf.experimental.ExtensionType):
    _raw_context: tf.Tensor
    log_n: int
    num_slots: int
    two_n: int
    main_moduli: tf.Tensor
    level: int
    aux_moduli: tf.Tensor
    plaintext_modulus: int
    noise_variance: int
    noise_bits: int
    scaling_factor: int
    mul_depth_supported: int
    seed: str

    def __init__(
        self,
        _raw_context,
        log_n,
        main_moduli,
        aux_moduli,
        plaintext_modulus,
        noise_variance,
        scaling_factor,
        mul_depth_supported,
        seed,
    ):
        self._raw_context = _raw_context
        self.log_n = log_n
        self.num_slots = 2**log_n
        self.two_n = 2 ** (log_n + 1)
        self.main_moduli = main_moduli
        self.level = len(main_moduli)
        self.aux_moduli = aux_moduli
        self.plaintext_modulus = plaintext_modulus
        self.noise_variance = noise_variance
        if self.noise_variance % 2 == 0:
            self.noise_bits = self.noise_variance.bit_length()
        else:
            self.noise_bits = self.noise_variance.bit_length() + 1
        self.scaling_factor = scaling_factor
        self.mul_depth_supported = mul_depth_supported
        self.seed = seed


def mod_reduce_context64(context):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    assert context.mul_depth_supported > 0, "Not enough multiplication primes."

    smaller_context = shell_ops.modulus_reduce_context64(context._raw_context)

    mod_reduced = ShellContext64(
        _raw_context=smaller_context,
        log_n=context.log_n,
        main_moduli=context.main_moduli[:-1],
        aux_moduli=context.aux_moduli,
        plaintext_modulus=context.plaintext_modulus,
        noise_variance=context.noise_variance,
        scaling_factor=context.scaling_factor,
        mul_depth_supported=context.mul_depth_supported - 1,
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
    mul_depth_supported=0,
    seed="",
):
    if len(seed) > 64:
        raise ValueError("Seed must be at most 64 characters long.")
    seed = seed.ljust(64)

    shell_context = shell_ops.context_import64(
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        seed=seed,
    )

    return ShellContext64(
        _raw_context=shell_context,
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        scaling_factor=scaling_factor,
        mul_depth_supported=mul_depth_supported,
        seed=seed,
    )
