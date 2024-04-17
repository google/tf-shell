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
import math
import random


class ShellContext64(object):
    def __init__(
        self,
        shell_context,
        log_n,
        main_moduli,
        aux_moduli,
        plaintext_modulus,
        noise_variance,
        scaling_factor,  # The field version of fixed point fractional bits.
        mul_depth_supported,
        seed,
    ):
        self._raw_context = shell_context
        self.log_n = log_n
        self.num_slots = 2**log_n
        self.two_n = 2 ** (self.log_n + 1)
        self.main_moduli = main_moduli
        self.aux_moduli = aux_moduli
        self.plaintext_modulus = plaintext_modulus
        self.noise_variance = noise_variance
        if noise_variance % 2 == 0:
            self.noise_bits = noise_variance.bit_length()
        else:
            self.noise_bits = noise_variance.bit_length() + 1
        self.scaling_factor = scaling_factor
        self.mul_depth_supported = mul_depth_supported
        self.seed = seed

    @property
    def level(self):
        return len(self.main_moduli)

    @property
    def Q(self):
        if not hasattr(self, "_Q"):
            self._Q = 1
            for x in self.main_moduli:
                self._Q *= x
        return self._Q

    def __lt__(self, other):
        return self.level < other.level

    def __le__(self, other):
        return self.level <= other.level

    def __gt__(self, other):
        return self.level > other.level

    def __ge__(self, other):
        return self.level >= other.level

    def __eq__(self, other):
        return (
            self.log_n == other.log_n
            and self.main_moduli == other.main_moduli
            and self.aux_moduli == other.aux_moduli
            and self.plaintext_modulus == other.plaintext_modulus
            and self.noise_variance == other.noise_variance
            and self.seed == other.seed
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (
            hash(tuple(self.main_moduli))
            ^ hash(tuple(self.aux_moduli))
            ^ hash(self.plaintext_modulus)
            ^ hash(self.noise_variance)
            ^ hash(self.seed)
        )

    def get_mod_reduced(self):
        assert self.mul_depth_supported > 0, "Not enough multiplication primes."

        if hasattr(self, "_mod_reduced"):
            return self._mod_reduced

        smaller_context = shell_ops.modulus_reduce_context64(self._raw_context)

        self._mod_reduced = ShellContext64(
            shell_context=smaller_context,
            log_n=self.log_n,
            main_moduli=self.main_moduli[:-1],
            aux_moduli=self.aux_moduli,
            plaintext_modulus=self.plaintext_modulus,
            noise_variance=self.noise_variance,
            scaling_factor=self.scaling_factor,
            mul_depth_supported=self.mul_depth_supported - 1,
            seed=self.seed,
        )

        return self._mod_reduced


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
        shell_context=shell_context,
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        scaling_factor=scaling_factor,
        mul_depth_supported=mul_depth_supported,
        seed=seed,
    )
