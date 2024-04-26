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
import math
import tensorflow as tf
import tf_shell.python.ops.shell_ops as shell_ops
from tf_shell.python.shell_context import ShellContext64
from tf_shell.python.shell_key import ShellKey64
from tf_shell.python.shell_key import ShellRotationKey64


class ShellTensor64(object):
    is_tensor_like = True  # needed to pass tf.is_tensor, new as of TF 2.2+

    def __init__(
        self,
        value,
        context,
        underlying_dtype,
        scaling_factor,
        is_enc,
        noise_bit_count,
    ):
        assert isinstance(
            value, tf.Tensor
        ), f"Should be variant tensor, instead got {type(value)}"

        assert (
            value.dtype is tf.variant
        ), f"Should be variant tensor, instead got {value.dtype}"

        assert isinstance(context, ShellContext64), f"Should be ShellContext64"

        self._raw = value
        self._context = context
        self._underlying_dtype = underlying_dtype
        self._scaling_factor = scaling_factor
        self._is_enc = is_enc

        self._noise_bit_count = noise_bit_count

    @property
    def shape(self):
        return [self._context.num_slots] + self._raw.shape

    @property
    def name(self):
        return self._raw.name

    @property
    def dtype(self):
        return self._raw.name

    @property
    def plaintext_dtype(self):
        return self._underlying_dtype

    @property
    def is_encrypted(self):
        return self._is_enc

    @property
    def noise_bits(self):
        return self._noise_bit_count + 1

    @property
    def level(self):
        return self._context.level

    @property
    def mul_depth_left(self):
        return self._context.mul_depth_supported

    def __getitem__(self, slice):
        slots = slice[0]
        if slots.start != None or slots.stop != None or slots.step != None:
            raise ValueError(
                f"ShellTensor does not support intra-slot slicing. Use `:` on the first dimension. Got {slice}"
            )
        return ShellTensor64(
            value=self._raw[slice[1:]],
            context=self._context,
            underlying_dtype=self._underlying_dtype,
            scaling_factor=self._scaling_factor,
            is_enc=self.is_encrypted,
            noise_bit_count=self._noise_bit_count,
        )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                result_raw = shell_ops.add_ct_ct64(
                    matched_self._raw, matched_other._raw
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw = shell_ops.add_ct_pt64(
                    matched_self._raw, matched_other._raw
                )
            elif not self.is_encrypted and other.is_encrypted:
                result_raw = shell_ops.add_ct_pt64(
                    matched_other._raw, matched_self._raw
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw = shell_ops.add_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw,
                    matched_other._raw,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=result_raw,
                context=matched_self._context,
                underlying_dtype=self._underlying_dtype,
                scaling_factor=self._scaling_factor,
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )

        elif isinstance(other, tf.Tensor):
            if other.shape == (1,) or other.shape == ():
                # In the special case of scalar addition, instead of padding
                # with zeros replicate the scalar across all slots.
                other = tf.broadcast_to(other, (self._context.num_slots, 1))

            # Lift tensorflow tensor to shell tensor with the same scaling
            # factor as self and attempt the addition again.
            so = to_shell_plaintext(other, self._context)
            return self + so

        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the subtraction again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(f"Unsupported type for addition. Got {type(other)}.")

            return self + tf_other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                result_raw = shell_ops.sub_ct_ct64(
                    matched_self._raw, matched_other._raw
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw = shell_ops.sub_ct_pt64(
                    matched_self._raw, matched_other._raw
                )
            elif not self.is_encrypted and other.is_encrypted:
                negative_other = -matched_other
                result_raw = shell_ops.add_ct_pt64(
                    negative_other._raw, matched_self._raw
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw = shell_ops.sub_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw,
                    matched_other._raw,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=result_raw,
                context=matched_self._context,
                underlying_dtype=self._underlying_dtype,
                scaling_factor=self._scaling_factor,
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )
        elif isinstance(other, tf.Tensor):
            if other.shape == (1,) or other.shape == ():
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots.
                other = tf.broadcast_to(other, (self._context.num_slots, 1))

            # Lift tensorflow tensor to shell tensor with the same scaling
            # factor as self and attempt the subtraction again.
            shell_other = to_shell_plaintext(other, self._context)
            return self - shell_other
        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the subtraction again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for subtraction. Got {type(other)}."
                )
            return self - tf_other

    def __rsub__(self, other):
        if isinstance(other, tf.Tensor):
            if other.shape == (1,) or other.shape == ():
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots.
                other = tf.broadcast_to(other, (self._context.num_slots, 1))

            # Import to a shell plaintext, which pads the first dimension with
            # zeros out to the number of slots.
            shell_other = to_shell_plaintext(other, self._context)

            # Match the shapes via broadcasting. This is after importing to
            # save NTTs.
            self_matched, other_matched = _match_shape(self, shell_other)

            if self_matched.is_encrypted:
                negative_self_matched = -self_matched
                raw_result = shell_ops.add_ct_pt64(
                    negative_self_matched._raw, other_matched._raw
                )
            else:
                raw_result = shell_ops.sub_pt_pt64(
                    self._context._raw_context, other_matched._raw, self_matched._raw
                )

            return ShellTensor64(
                value=raw_result,
                context=self._context,
                underlying_dtype=self._underlying_dtype,
                scaling_factor=self._scaling_factor,
                is_enc=self._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )
        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the the rsub again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for subtraction. Got {type(other)}."
                )
            return tf_other - self

    def __neg__(self):
        if self.is_encrypted:
            raw_result = shell_ops.neg_ct64(self._raw)
        else:
            raw_result = shell_ops.neg_pt64(self._context._raw_context, self._raw)

        return ShellTensor64(
            value=raw_result,
            context=self._context,
            underlying_dtype=self._underlying_dtype,
            scaling_factor=self._scaling_factor,
            is_enc=self._is_enc,
            noise_bit_count=self._noise_bit_count + 1,
        )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                raw_result = shell_ops.mul_ct_ct64(
                    matched_self._raw, matched_other._raw
                )
            elif self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_self._raw, matched_other._raw
                )
            elif not self.is_encrypted and other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_other._raw, matched_self._raw
                )
            elif not self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw,
                    matched_other._raw,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=raw_result,
                context=matched_self._context,
                underlying_dtype=self._underlying_dtype,
                scaling_factor=matched_self._scaling_factor**2,
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=matched_self._noise_bit_count
                + matched_other._noise_bit_count,
            )
        elif isinstance(other, tf.Tensor):
            # Multiplying by a scalar uses a special op which is more efficient
            # than the caller creating creating a ShellTensor the same
            # dimensions as self and multiplying.
            if other.shape == (1,) or other.shape == ():
                # Encode the other scalar tensor to the same scaling factor as
                # self.
                other = _encode_scaling(other, self._scaling_factor)

                if self.is_encrypted:
                    raw_result = shell_ops.mul_ct_tf_scalar64(
                        self._context._raw_context, self._raw, other
                    )
                else:
                    raw_result = shell_ops.mul_pt_tf_scalar64(
                        self._context._raw_context, self._raw, other
                    )

                return ShellTensor64(
                    value=raw_result,
                    context=self._context,
                    underlying_dtype=self._underlying_dtype,
                    scaling_factor=self._scaling_factor**2,
                    is_enc=self._is_enc,
                    noise_bit_count=self.noise_bits + self._context.noise_bits,
                )

            else:
                # Import the TensorFlow tensor to a shell plaintext and attempt
                # the multiplication again.
                shell_other = to_shell_plaintext(other, self._context)

                return self * shell_other

        else:
            # Try to import the unknown multiplicand to a TensorFlow tensor and
            # attempt the multiplication again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for multiplication. Got {type(other)}."
                )
            return self * tf_other

    def __rmul__(self, other):
        return self * other

    def get_mod_reduced(self):
        """Switches the ShellTensor to a new context with different moduli. If
        preserve_plaintext is True (default), the plaintext value will be
        maintained through the modulus switch. If preserve_plaintext is False,
        the plaintext will be divided by the ratio of the new and old moduli."""

        if hasattr(self, "_mod_reduced"):
            return self._mod_reduced

        # Switch to the new context and moduli.
        if self.is_encrypted:
            op = shell_ops.modulus_reduce_ct64
        else:
            op = shell_ops.modulus_reduce_pt64

        raw_result = op(
            self._context._raw_context,
            self._raw,
        )

        reduced_self = ShellTensor64(
            value=raw_result,
            context=self._context.get_mod_reduced(),
            underlying_dtype=self._underlying_dtype,
            scaling_factor=self._scaling_factor,
            is_enc=self._is_enc,
            noise_bit_count=self.noise_bits
            - self._context.main_moduli[-1].bit_length()
            + 1,
        )

        # Cache the result.
        self._mod_reduced = reduced_self

        return reduced_self


def _match_moduli_and_scaling(x, y):
    # Mod switch to the smaller modulus of the two.
    while x._context.level > y._context.level:
        x = x.get_mod_reduced()
    while x._context.level < y._context.level:
        y = y.get_mod_reduced()

    # Match the scaling factors.
    # First make sure the scaling factors are compatible.
    frac = x._scaling_factor / y._scaling_factor
    if abs(frac - int(frac)) != 0:
        raise ValueError(
            f"Scaling factors must be compatible. Got {x._scaling_factor} and {y._scaling_factor}"
        )

    while x._scaling_factor > y._scaling_factor:
        y = y * x._scaling_factor
    while x._scaling_factor < y._scaling_factor:
        x = x * y._scaling_factor

    x, y = _match_shape(x, y)

    return x, y


def _match_shape(x, y):
    # Match the shape of x and y via broadcasting.
    if tf.size(x._raw) > tf.size(y._raw):
        y = ShellTensor64(
            value=tf.broadcast_to(y._raw, tf.shape(x._raw)),
            context=y._context,
            underlying_dtype=y._underlying_dtype,
            scaling_factor=y._scaling_factor,
            is_enc=y._is_enc,
            noise_bit_count=y._noise_bit_count,
        )
    elif tf.size(x._raw) < tf.size(y._raw):
        x = ShellTensor64(
            value=tf.broadcast_to(x._raw, tf.shape(y._raw)),
            context=y._context,
            underlying_dtype=y._underlying_dtype,
            scaling_factor=y._scaling_factor,
            is_enc=y._is_enc,
            noise_bit_count=y._noise_bit_count,
        )

    return x, y


def _get_shell_dtype_from_underlying(type):
    if type in [tf.float32, tf.float64]:
        return tf.int64
    elif type in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.uint64
    elif type in [tf.int8, tf.int16, tf.int32, tf.int64]:
        return tf.int64
    else:
        raise ValueError(f"Unsupported type {type}")


def _encode_scaling(tf_tensor, scaling_factor=1):
    if tf_tensor.dtype in [tf.float32, tf.float64]:
        return tf.cast(tf.round(tf_tensor * scaling_factor), tf.int64)
    elif tf_tensor.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        # Pass unsigned datatypes to shell as uint64.
        return tf.cast(tf_tensor, tf.uint64)
    elif tf_tensor.dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
        # Pass signed datatypes to shell as int64.
        return tf.cast(tf_tensor, tf.int64)
    else:
        raise ValueError(f"Unsupported dtype {tf_tensor.dtype}")


def _decode_scaling(scaled_tensor, output_dtype, scaling_factor):
    if output_dtype in [tf.float32, tf.float64]:
        assert scaled_tensor.dtype == tf.int64
        return tf.cast(scaled_tensor, output_dtype) / scaling_factor
    elif output_dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        assert scaled_tensor.dtype == tf.uint64
        return tf.cast(scaled_tensor, output_dtype)
    elif output_dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
        assert scaled_tensor.dtype == tf.int64
        return tf.cast(scaled_tensor, output_dtype)
    else:
        raise ValueError(f"Unsupported dtype {output_dtype}")


def to_shell_plaintext(tensor, context):
    """Converts a Tensorflow tensor to a ShellTensor which holds a plaintext.
    Under the hood, this means encoding the Tensorflow tensor to a BGV style
    polynomial representation with the sign and scaling factor encoded as
    appropriate."""
    assert isinstance(
        context, ShellContext64
    ), f"Context must be a ShellContext64, instead got {type(context)}"

    if isinstance(tensor, ShellTensor64):
        if tensor.is_encrypted:
            raise ValueError(
                "Cannot use to_plaintext() on an encrypted ShellTensor. Use to_tensorflow() with a key to decrypt."
            )
        return tensor  # Do nothing, already a shell plaintext.

    elif isinstance(tensor, tf.Tensor):
        # Shell tensor represents floats as integers * scaling_factor.
        scaled_tensor = _encode_scaling(tensor, context.scaling_factor)

        # Pad the tensor to the correct number of slots.
        if scaled_tensor.shape[0] > context.num_slots:
            raise ValueError(
                f"Tensor first dimension is too large. Maximum is {context.num_slots}, got {scaled_tensor.shape[0]}."
            )
        elif scaled_tensor.shape[0] < context.num_slots:
            padding = [[0, context.num_slots - scaled_tensor.shape[0]]] + [
                [0, 0] for _ in range(len(scaled_tensor.shape) - 1)
            ]
            scaled_tensor = tf.pad(scaled_tensor, padding)

        return ShellTensor64(
            value=shell_ops.polynomial_import64(context._raw_context, scaled_tensor),
            context=context,
            underlying_dtype=tensor.dtype,
            scaling_factor=context.scaling_factor,
            is_enc=False,
            noise_bit_count=context.noise_bits,
        )
    else:
        try:
            return to_shell_plaintext(tf.convert_to_tensor(tensor), context)
        except:
            raise ValueError(f"Cannot convert to ShellTensor64. Got {type(tensor)}.")


def to_encrypted(x, key, context=None):
    """Encrypts a plaintext tensor or ShellTensor using the provided key. If
    the input is a Tensorflow tensor, a context must also be provided. If the
    input is a ShellTensor, the context is ignored and the context stored
    in the input ShellTensor is used."""
    if not isinstance(key, ShellKey64):
        raise ValueError("Key must be a ShellKey64")

    if isinstance(x, ShellTensor64):
        if x._is_enc:
            return x  # Do nothing, already encrypted.
        else:
            return ShellTensor64(
                value=shell_ops.encrypt64(
                    x._context._raw_context,
                    key._raw_key,
                    x._raw,
                ),
                context=x._context,
                underlying_dtype=x._underlying_dtype,
                scaling_factor=x._scaling_factor,
                is_enc=True,
                noise_bit_count=x._noise_bit_count,
            )
    else:
        if not isinstance(context, ShellContext64):
            raise ValueError(
                "ShellContext64 must be provided when encrypting anything other than a ShellTensor64."
            )

        # Encode to a shell plaintext using the provided context and call
        # ourself to encrypt.
        return to_encrypted(to_shell_plaintext(x, context), key)


def to_tensorflow(s_tensor, key=None):
    """Converts a ShellTensor to a Tensorflow tensor. If the ShellTensor is
    encrypted, a key must be provided to decrypt it. If the ShellTensor is
    plaintext, the key is ignored."""
    assert isinstance(
        s_tensor, ShellTensor64
    ), f"Should be ShellTensor, instead got {type(s_tensor)}"

    # Find out what dtype shell thinks the plaintext is.
    shell_dtype = _get_shell_dtype_from_underlying(s_tensor._underlying_dtype)

    if s_tensor.is_encrypted:
        if not isinstance(key, ShellKey64):
            raise ValueError(
                "Key must be provided to decrypt an encrypted ShellTensor."
            )

        # Mod reduce the key to match the level of the ciphertext.
        while key.level > s_tensor._context.level:
            key = key.get_mod_reduced()

        # Decrypt op returns a tf Tensor.
        tf_tensor = shell_ops.decrypt64(
            s_tensor._context._raw_context,
            key._raw_key,
            s_tensor._raw,
            dtype=shell_dtype,
        )

    else:
        # Convert from polynomial representation to plaintext tensorflow tensor.
        # Always convert to int64, then handle the fixed point as appropriate.
        tf_tensor = shell_ops.polynomial_export64(
            s_tensor._context._raw_context,
            s_tensor._raw,
            dtype=shell_dtype,
        )

    # Shell tensor represents floats as integers * scaling_factor.
    return _decode_scaling(
        tf_tensor,
        s_tensor._underlying_dtype,
        s_tensor._scaling_factor,
    )


def roll(x, shift, rotation_key):
    if isinstance(x, ShellTensor64):
        if not isinstance(rotation_key, ShellRotationKey64):
            raise ValueError(
                f"Rotation key must be provided. Instead saw {rotation_key}."
            )

        if not x._is_enc:
            raise ValueError("Unencrypted ShellTensor rotation not supported yet.")

        # Get the correct rotation key for the level of this ciphertext.
        raw_rotation_key = rotation_key._get_key_at_level(x._context.level)

        shift = tf.cast(shift, tf.int64)

        return ShellTensor64(
            value=shell_ops.roll64(raw_rotation_key, x._raw, shift),
            context=x._context,
            underlying_dtype=x._underlying_dtype,
            scaling_factor=x._scaling_factor,
            is_enc=True,
            noise_bit_count=x._noise_bit_count + 6,  # TODO correct?
        )
    elif isinstance(x, tf.Tensor):
        return tf.roll(x, shift)
    else:
        raise ValueError(f"Unsupported type for reduce_sum. Got {type(x)}.")


def reduce_sum(x, axis, rotation_key=None):
    if isinstance(x, ShellTensor64):
        if not x._is_enc:
            raise ValueError("Unencrypted ShellTensor reduce_sum not supported yet.")

        # Check axis is a scalar
        if isinstance(axis, tf.Tensor) and not axis.shape != []:
            raise ValueError("Only scalar `axis` is supported.")

        if axis == 0:
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    f"Rotation key must be provided to reduce_sum over the first axis. Instead saw {rotation_key}."
                )

            # Get the correct rotation key for the level of this ciphertext.
            raw_rotation_key = rotation_key._get_key_at_level(x._context.level)

            # reduce sum does log2(num_slots) rotations and additions.
            # TODO: add noise from rotations?
            result_noise_bits = (
                x._noise_bit_count + x._context.num_slots.bit_length() + 1,
            )

            return ShellTensor64(
                value=shell_ops.reduce_sum_by_rotation64(x._raw, raw_rotation_key),
                context=x._context,
                underlying_dtype=x._underlying_dtype,
                scaling_factor=x._scaling_factor,
                is_enc=True,
                noise_bit_count=result_noise_bits,
            )
        else:
            if axis >= len(x.shape):
                raise ValueError("Axis greater than number of dimensions")

            result_noise_bits = x._noise_bit_count + x.shape[axis].bit_length() + 1

            return ShellTensor64(
                value=shell_ops.reduce_sum64(x._raw, axis),
                context=x._context,
                underlying_dtype=x._underlying_dtype,
                scaling_factor=x._scaling_factor,
                is_enc=True,
                noise_bit_count=result_noise_bits,
            )
    elif isinstance(x, tf.Tensor):
        return tf.reduce_sum(x, axis)
    else:
        raise ValueError(f"Unsupported type for reduce_sum. Got {type(x)}.")


def matmul(x, y, rotation_key=None):
    """Matrix multiplication is specialized to whether the operands are
    plaintext or ciphertext.

    matmul(ciphertext, plaintext) works as in Tensorflow.

    matmul(plaintext, ciphertext) in tf-shell has slightly different semantics
    than plaintext / Tensorflow. tf-shell affects top and bottom halves
    independently, as well as the first dimension repeating the sum of either
    the halves."""
    if isinstance(x, ShellTensor64) and isinstance(y, tf.Tensor):
        if x._underlying_dtype != y.dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x._underlying_dtype} and {y.dtype}"
            )

        if x.mul_depth_left <= 0:
            raise ValueError(
                "Insufficient multiplication depth remaining to perform matmul."
            )

        # Encode the plaintext y to the same scaling factor as x.
        scaled_y = _encode_scaling(y, x._scaling_factor)

        # Noise grows from one multiplication then a sum over that dimension.
        multiplication_noise = x.noise_bits + x._context.noise_bits
        reduce_sum_noise = multiplication_noise + x.shape[1].bit_length()

        return ShellTensor64(
            value=shell_ops.mat_mul_ct_pt64(
                x._context._raw_context,
                x._raw,
                scaled_y,
            ),
            context=x._context,
            underlying_dtype=x._underlying_dtype,
            scaling_factor=x._scaling_factor**2,
            is_enc=True,
            noise_bit_count=reduce_sum_noise,
        )

    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
        if not isinstance(rotation_key, ShellRotationKey64):
            raise ValueError(
                f"Rotation key must be provided to matmul pt*ct. Instead saw {rotation_key}."
            )

        if x.dtype != y._underlying_dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x.dtype} and {y._underlying_dtype}"
            )

        if y.mul_depth_left <= 0:
            raise ValueError(
                "Insufficient multiplication depth remaining to perform matmul."
            )

        # Encode the plaintext x to the same scaling factor as y.
        scaled_x = _encode_scaling(x, y._scaling_factor)

        # Get the correct rotation key for the level of y.
        raw_rotation_key = rotation_key._get_key_at_level(y._context.level)

        # Noise grows from doing one multiplication then a reduce_sum operation
        # over the outer (ciphertext) dimension. dimension. The noise from the
        # reduce_sum is a rough estimate that works for slots = 2**11.
        multiplication_noise = y._noise_bit_count + 1
        rotation_noise = multiplication_noise + 60
        reduce_sum_noise = rotation_noise + y._context.num_slots.bit_length()

        return ShellTensor64(
            value=shell_ops.mat_mul_pt_ct64(
                y._context._raw_context,
                raw_rotation_key,
                scaled_x,
                y._raw,
            ),
            context=y._context,
            underlying_dtype=y._underlying_dtype,
            scaling_factor=y._scaling_factor**2,
            is_enc=True,
            noise_bit_count=reduce_sum_noise,
        )

    elif isinstance(x, ShellTensor64) and isinstance(y, ShellTensor64):
        return NotImplementedError

    elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        return tf.matmul(x, y)

    else:
        raise ValueError(
            f"Unsupported types for matmul. Got {type(x)} and {type(y)}. If multiplying by a plaintext, pass it as a plain TensorFlow tensor, not a ShellTensor."
        )


def expand_dims(x, axis=-1):
    if isinstance(x, ShellTensor64):
        # Perform some checks on the axis.
        if axis == 0:
            raise ValueError(
                "Cannot expand dims at axis 0 for ShellTensor64, this is the batching dimension."
            )
        return ShellTensor64(
            value=shell_ops.expand_dims_variant(x._raw, axis),
            context=x._context,
            underlying_dtype=x._underlying_dtype,
            scaling_factor=x._scaling_factor,
            is_enc=x._is_enc,
            noise_bit_count=x._noise_bit_count,
        )
    elif isinstance(x, tf.Tensor):
        return tf.expand_dims(x, axis)
    else:
        raise ValueError("Unsupported type for expand_dims")


def reshape(x, shape):
    if isinstance(x, ShellTensor64):
        # Perform some checks on the new shape.
        if shape[0] != x._context.num_slots:
            raise ValueError(
                "Cannot reshape axis 0 for ShellTensor64, this is the batching dimension."
            )
        return ShellTensor64(
            value=tf.reshape(x._raw, shape[1:]),
            context=x._context,
            underlying_dtype=x._underlying_dtype,
            scaling_factor=x._scaling_factor,
            is_enc=x._is_enc,
            noise_bit_count=x._noise_bit_count,
        )
    elif isinstance(x, tf.Tensor):
        return tf.reshape(x, shape)
    else:
        raise ValueError("Unsupported type for expand_dims")
