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

test_fxp_fractional_bits = [0, 1, 2, 3, 4]
test_dtypes = [
    tf.int8,
    tf.uint8,
    tf.int16,
    tf.uint16,
    tf.int32,
    tf.uint32,
    tf.int64,
    tf.uint64,
    tf.float32,
    tf.float64,
]


class TestContext:
    def __init__(self, outer_shape, log_slots, main_moduli, plaintext_modulus):
        self.outer_shape = outer_shape
        self.log_slots = log_slots
        self.slots = 2**log_slots

        self.plaintext_modulus = plaintext_modulus

        self.shell_context = shell_tensor.create_context64(
            log_n=log_slots,
            main_moduli=main_moduli,
            aux_moduli=[],
            plaintext_modulus=plaintext_modulus,
            noise_variance=4,
            seed="",
        )

        self.key = shell_tensor.create_key64(self.shell_context)

        self.rotation_key = shell_tensor.create_rotation_key64(
            self.shell_context, self.key
        )


test_contexts = [
    TestContext(
        outer_shape=[3, 2, 3],
        log_slots=11,
        main_moduli=[8556589057, 8388812801],
        plaintext_modulus=40961,
    ),
]


def get_bounds_for_n_muls(dtype, plaintext_modulus, num_frac_bits, num_muls):
    """Returns a safe range for plaintext values when doing a given number of
    multiplications. The range is determined by both the plaintext modulus and
    the datatype."""
    max_fractional_bits = 2**num_muls * num_frac_bits
    max_fractional_value = 2**max_fractional_bits

    # Make sure not to exceed the range of the dtype.
    min_plaintext_dtype = -math.floor(abs(dtype.min) ** (1.0 / (num_muls + 1)))
    min_plaintext_dtype = math.floor(min_plaintext_dtype / max_fractional_value)
    max_plaintext_dtype = math.floor(dtype.max ** (1.0 / (num_muls + 1)))
    max_plaintext_dtype = math.floor(max_plaintext_dtype / max_fractional_value)

    # Make sure not to exceed the range of the plaintext modulus.
    if dtype.is_unsigned:
        min_plaintext_modulus = 0
        # max_plaintext_modulus = int(plaintext_modulus - 1)
        max_plaintext_modulus = plaintext_modulus ** (1.0 / (num_muls + 1))
        max_plaintext_modulus = math.floor(max_plaintext_modulus)
        max_plaintext_modulus = math.floor(max_plaintext_modulus / max_fractional_value)
    else:
        range = math.floor(plaintext_modulus / 2)
        range **= 1.0 / (num_muls + 1)
        range = math.floor(range / max_fractional_value)
        min_plaintext_modulus = -range
        max_plaintext_modulus = range

    min_val = max(min_plaintext_dtype, min_plaintext_modulus)
    max_val = min(max_plaintext_dtype, max_plaintext_modulus)

    return min_val, max_val


def get_bounds_for_n_adds(dtype, plaintext_modulus, num_frac_bits, num_adds):
    """Returns a safe range for plaintext values when doing a given number of
    additions."""
    max_fractional_bits = num_frac_bits
    max_fractional_value = 2**max_fractional_bits

    # Make sure not to exceed the range of the dtype.
    min_plaintext_dtype = math.ceil(dtype.min / (num_adds + 1))
    min_plaintext_dtype = math.ceil(min_plaintext_dtype / max_fractional_value)
    max_plaintext_dtype = math.floor(dtype.max / (num_adds + 1))
    max_plaintext_dtype = math.floor(max_plaintext_dtype / max_fractional_value)

    # Make sure not to exceed the range of the plaintext modulus.
    if dtype.is_unsigned:
        min_plaintext_modulus = 0
        # max_plaintext_modulus = int(plaintext_modulus - 1)
        max_plaintext_modulus = (plaintext_modulus - 1) / (num_adds + 1)
        max_plaintext_modulus = math.floor(max_plaintext_modulus)
        max_plaintext_modulus = math.floor(max_plaintext_modulus / max_fractional_value)
    else:
        range = math.floor(plaintext_modulus / 2)
        range /= num_adds + 1
        range = math.floor(range / max_fractional_value)
        min_plaintext_modulus = -range
        max_plaintext_modulus = range

    min_val = max(min_plaintext_dtype, min_plaintext_modulus)
    max_val = min(max_plaintext_dtype, max_plaintext_modulus)

    return min_val, max_val


def uniform_for_n_adds(dtype, test_context, num_fxp_frac_bits, num_adds):
    """Returns a random tensor with values in the range of the datatype and
    plaintext modulus. The elements support n additions without overflowing
    either the datatype and plaintext modulus. Floating point datatypes return
    fractional values at the appropriate quantization."""
    min_val, max_val = get_bounds_for_n_adds(
        dtype, test_context.plaintext_modulus, num_fxp_frac_bits, num_adds
    )

    if max_val < 2 ** (-num_fxp_frac_bits - 1):
        return None

    if dtype.is_floating:
        min_val *= 2**num_fxp_frac_bits
        max_val *= 2**num_fxp_frac_bits

    shape = test_context.outer_shape.copy()
    shape.insert(0, test_context.slots)

    rand = tf.random.uniform(
        shape,
        dtype=tf.int64,
        maxval=max_val,
        minval=min_val,
    )

    rand = tf.cast(rand, dtype)
    if dtype.is_floating:
        rand /= 2**num_fxp_frac_bits

    return rand


def uniform_for_n_muls(
    dtype, test_context, num_fxp_frac_bits, num_muls, shape=None, subsequent_adds=0
):
    """Returns a random tensor with values in the range of the datatype and
    plaintext modulus. The elements support n additions without overflowing
    either the datatype and plaintext modulus. Floating point datatypes return
    fractional values at the appropriate quantization.
    """
    min_val, max_val = get_bounds_for_n_muls(
        dtype, test_context.plaintext_modulus, num_fxp_frac_bits, num_muls
    )

    min_val = math.floor(min_val / (subsequent_adds + 1))
    max_val = math.floor(max_val / (subsequent_adds + 1))

    if max_val < 2 ** (-num_fxp_frac_bits - 1):
        return None

    if dtype.is_floating:
        min_val *= 2**num_fxp_frac_bits
        max_val *= 2**num_fxp_frac_bits

    if shape is None:
        shape = test_context.outer_shape.copy()
        shape.insert(0, test_context.slots)

    rand = tf.random.uniform(
        shape,
        dtype=tf.int64,
        maxval=max_val,
        minval=min_val,
    )

    rand = tf.cast(rand, dtype)
    if dtype.is_floating:
        rand /= 2**num_fxp_frac_bits

    return rand
