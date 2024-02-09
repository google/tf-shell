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
import shell_tensor
import math


class TestContext:
    def __init__(
        self,
        outer_shape,
        plaintext_dtype,
        log_n,
        main_moduli,
        aux_moduli,
        plaintext_modulus,
        noise_variance=8,
        scaling_factor=1,
        mul_depth_supported=0,
        seed="",
    ):
        self.outer_shape = outer_shape
        self.plaintext_dtype = plaintext_dtype

        self.shell_context = shell_tensor.create_context64(
            log_n=log_n,
            main_moduli=main_moduli,
            aux_moduli=aux_moduli,
            plaintext_modulus=plaintext_modulus,
            noise_variance=noise_variance,
            scaling_factor=scaling_factor,
            mul_depth_supported=mul_depth_supported,
            seed=seed,
        )

        self.key = shell_tensor.create_key64(self.shell_context)

    @property
    def rotation_key(self):
        # Rotation keys are slow to generate. Only generate them on demand and
        # cache them.
        if not hasattr(self, "_rotation_key"):
            self._rotation_key = shell_tensor.create_rotation_key64(
                self.shell_context, self.key
            )
        return self._rotation_key

    def __str__(self):
        return f"log_n {self.shell_context.log_n}, plaintext_modulus {self.shell_context.plaintext_modulus}, plaintext_dtype {self.plaintext_dtype}, scaling_factor {self.shell_context.scaling_factor}"


def get_bounds_for_n_adds(test_context, num_adds):
    """Returns a safe range for plaintext values when doing a given number of
    additions."""
    dtype = test_context.plaintext_dtype
    plaintext_modulus = test_context.shell_context.plaintext_modulus
    scaling_factor_squared = test_context.shell_context.scaling_factor**2

    # Make sure not to exceed the range of the dtype.
    min_plaintext_dtype = math.ceil(dtype.min / scaling_factor_squared)
    min_plaintext_dtype = math.ceil(min_plaintext_dtype / (num_adds + 1))
    max_plaintext_dtype = dtype.max // scaling_factor_squared
    max_plaintext_dtype //= num_adds + 1

    # Make sure not to exceed the range of the plaintext modulus.
    if dtype.is_unsigned:
        min_plaintext_modulus = 0
        max_plaintext_modulus = plaintext_modulus // scaling_factor_squared
        max_plaintext_modulus //= num_adds + 1
    else:
        range = plaintext_modulus // 2
        range //= scaling_factor_squared
        range /= num_adds + 1
        min_plaintext_modulus = -range
        max_plaintext_modulus = range

    min_val = max(min_plaintext_dtype, min_plaintext_modulus)
    max_val = min(max_plaintext_dtype, max_plaintext_modulus)

    if min_val > max_val:
        raise ValueError(
            f"min_val {min_val} > max_val {max_val} for num_adds {num_adds}"
        )

    return min_val, max_val


def get_bounds_for_n_muls(test_context, num_muls):
    """Returns a safe range for plaintext values when doing a given number of
    multiplications. The range is determined by both the plaintext modulus and
    the datatype."""
    dtype = test_context.plaintext_dtype
    plaintext_modulus = test_context.shell_context.plaintext_modulus
    scaling_factor_squared = test_context.shell_context.scaling_factor**2

    # Make sure not to exceed the range of the dtype.
    min_plaintext_dtype = math.ceil(dtype.min / scaling_factor_squared)
    min_plaintext_dtype = -math.floor(
        abs(min_plaintext_dtype) ** (1.0 / (num_muls + 1))
    )
    max_plaintext_dtype = dtype.max // scaling_factor_squared
    max_plaintext_dtype **= 1.0 / (num_muls + 1)

    # Make sure not to exceed the range of the plaintext modulus.
    if dtype.is_unsigned:
        min_plaintext_modulus = 0
        max_plaintext_modulus = plaintext_modulus // scaling_factor_squared
        max_plaintext_modulus **= 1.0 / (num_muls + 1)
    else:
        range = plaintext_modulus // 2
        range //= scaling_factor_squared
        range **= 1.0 / (num_muls + 1)
        min_plaintext_modulus = -range
        max_plaintext_modulus = range

    min_val = max(min_plaintext_dtype, min_plaintext_modulus)
    max_val = min(max_plaintext_dtype, max_plaintext_modulus)

    if min_val > max_val:
        raise ValueError(
            f"min_val {min_val} > max_val {max_val} for num_muls {num_adds}"
        )

    return min_val, max_val


def uniform_for_n_adds(test_context, num_adds, shape=None):
    """Returns a random tensor with values in the range of the datatype and
    plaintext modulus. The elements support n additions without overflowing
    either the datatype and plaintext modulus. Floating point datatypes return
    fractional values at the appropriate quantization."""
    min_val, max_val = get_bounds_for_n_adds(test_context, num_adds)

    scaling_factor = test_context.shell_context.scaling_factor

    if max_val < 1 / scaling_factor:
        raise ValueError(
            f"Available plaintext range for the given number of additions [{min_val}, {max_val}] is too small. Must be larger than {1/scaling_factor}."
        )

    if test_context.plaintext_dtype.is_floating:
        min_val *= scaling_factor
        max_val *= scaling_factor

    if shape is None:
        shape = test_context.outer_shape.copy()
        shape.insert(0, test_context.shell_context.num_slots)

    rand = tf.random.uniform(
        shape,
        dtype=tf.int64,
        minval=math.ceil(min_val),
        maxval=math.floor(max_val),
    )

    if test_context.plaintext_dtype.is_floating:
        rand /= scaling_factor

    rand = tf.cast(rand, test_context.plaintext_dtype)

    return rand


def uniform_for_n_muls(test_context, num_muls, shape=None, subsequent_adds=0):
    """Returns a random tensor with values in the range of the datatype and
    plaintext modulus. The elements support n additions without overflowing
    either the datatype and plaintext modulus. Floating point datatypes return
    fractional values at the appropriate quantization.
    """
    scaling_factor = test_context.shell_context.scaling_factor
    if scaling_factor > 1 and num_muls > test_context.shell_context.mul_depth_supported:
        raise ValueError("Number of multiplications not supported by context.")

    min_val, max_val = get_bounds_for_n_muls(test_context, num_muls)

    min_val = min_val / (subsequent_adds + 1)
    max_val = max_val / (subsequent_adds + 1)

    if max_val < 1 / scaling_factor:
        raise ValueError(
            f"Available plaintext range for the given number of multiplications [{min_val}, {max_val}] is too small. Must be larger than {1/scaling_factor}."
        )

    if test_context.plaintext_dtype.is_floating:
        min_val *= scaling_factor
        max_val *= scaling_factor

    if shape is None:
        shape = test_context.outer_shape.copy()
        shape.insert(0, test_context.shell_context.num_slots)

    rand = tf.random.uniform(
        shape,
        dtype=tf.int64,
        minval=math.ceil(min_val),
        maxval=math.floor(max_val),
    )

    if test_context.plaintext_dtype.is_floating:
        rand /= scaling_factor

    rand = tf.cast(rand, test_context.plaintext_dtype)

    return rand
