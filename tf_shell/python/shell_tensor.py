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
import tf_shell.shell as raw_bindings
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
        self._is_enc = is_enc

        self._noise_bit_count = noise_bit_count
        self._mod_reduced = None

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

    def __getitem__(self, slice):
        slots = slice[0]
        if slots.start != None or slots.stop != None or slots.step != None:
            raise ValueError(
                f"ShellTensor does not support intra-slot slicing. Be sure to use `:` on the first dimension. Got {slice}"
            )
        return ShellTensor64(
            value=self._raw[slice[1:]],
            context=self._context,
            underlying_dtype=self._underlying_dtype,
            is_enc=self.is_encrypted,
            noise_bit_count=self._noise_bit_count,
        )

    def get_encrypted(self, key):
        if not isinstance(key, ShellKey64):
            raise ValueError("Key must be a ShellKey64")

        if self._is_enc:
            return self
        else:
            return ShellTensor64(
                value=shell_ops.encrypt64(
                    self._context._raw_context,
                    key._raw_key,
                    self._raw,
                ),
                context=self._context,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                noise_bit_count=self._noise_bit_count,
            )

    def get_decrypted(self, key=None):
        if not self._is_enc:
            return from_shell_tensor(self)
        else:
            if not isinstance(key, ShellKey64):
                raise ValueError("Key must be a ShellKey64")

            # Find out what dtype shell thinks the plaintext is.
            shell_dtype = _get_shell_dtype_from_underlying(self._underlying_dtype)

            while key.level > self._context.level:
                key = key.get_mod_reduced()

            # Decrypt op returns a tf Tensor.
            tf_tensor = shell_ops.decrypt64(
                self._context._raw_context,
                key._raw_key,
                self._raw,
                dtype=shell_dtype,
            )

            # Convert back to the underlying dtype.
            return _decode_scaling(
                tf_tensor,
                self._underlying_dtype,
                self._context.scaling_factor * self._context.scaling_factor,
            )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            # Mod switch to the smaller modulus of the two.
            matched_self = self
            matched_other = other
            while matched_self._context.level > matched_other._context.level:
                matched_self = matched_self.get_mod_reduced()
            while matched_self._context.level < matched_other._context.level:
                matched_other = matched_other.get_mod_reduced()

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
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )

        elif isinstance(other, tf.Tensor):
            # TODO(jchoncholas): Adding a scalar uses a special op that is
            # more efficient.
            if other.shape == []:
                raise ValueError("Scalar addition not yet implemented.")

            # Lift tensorflow tensor to shell tensor with the same scaling
            # factor as self.
            so = to_shell_tensor(self._context, other)
            return self + so

        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ShellTensor64):
            # Mod switch to the smaller modulus of the two.
            matched_self = self
            matched_other = other
            while matched_self._context.level > matched_other._context.level:
                matched_self = matched_self.get_mod_reduced()
            while matched_self._context.level < matched_other._context.level:
                matched_other = matched_other.get_mod_reduced()

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
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )
        elif isinstance(other, tf.Tensor):
            # TODO(jchoncholas): Subtracting a scalar uses a special op that is
            # more efficient.
            if other.shape == []:
                raise ValueError("Scalar subtraction not yet implemented.")

            # Lift tensorflow tensor to shell tensor with the same number
            # fractional bits as self.
            so = to_shell_tensor(self._context, other)
            return self - so
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, tf.Tensor):
            shell_other = to_shell_tensor(self._context, other)

            if self.is_encrypted:
                negative_self = -self
                raw_result = shell_ops.add_ct_pt64(negative_self._raw, shell_other._raw)
            else:
                raw_result = shell_ops.sub_pt_pt64(
                    self._context._raw_context, shell_other._raw, self._raw
                )

            return ShellTensor64(
                value=raw_result,
                context=self._context,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc,
                noise_bit_count=self._noise_bit_count + 1,
            )
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_encrypted:
            raw_result = shell_ops.neg_ct64(self._raw)
        else:
            raw_result = shell_ops.neg_pt64(self._context._raw_context, self._raw)

        return ShellTensor64(
            value=raw_result,
            context=self._context,
            underlying_dtype=self._underlying_dtype,
            is_enc=self._is_enc,
            noise_bit_count=self._noise_bit_count + 1,
        )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            # First mod switch to the smaller modulus of the two.
            matched_self = self
            matched_other = other
            while matched_self._context.level > matched_other._context.level:
                matched_self = matched_self.get_mod_reduced()
            while matched_self._context.level < matched_other._context.level:
                matched_other = matched_other.get_mod_reduced()

            # Next, reduce the scaling factor of each operand from 2x to 1x so
            # that after multiplication, the result is back to 2x the scaling
            # factor. This division is done by mod switching. Only required
            # when the scaling factor is greater than 1.
            if matched_self._context.scaling_factor > 1:
                matched_self = matched_self.get_mod_reduced(preserve_plaintext=False)
                matched_other = matched_other.get_mod_reduced(preserve_plaintext=False)

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
                is_enc=self._is_enc or other._is_enc,
                noise_bit_count=matched_self._noise_bit_count
                + matched_other._noise_bit_count,
            )
        elif isinstance(other, tf.Tensor):
            # Here we will take a shortcut since the other tensor has not yet
            # been encoded as a shell tensor. Naively, one would encode the
            # other tensor to the scaling factor**2, as usual, then reduce it
            # back down to 1x the scaling factor before multiplying. Instead,
            # here we lift the plaintext directly to 1x the scaling factor.
            # When the scaling factor is 1, this only performs a cast.
            single_scaled_other = _encode_scaling(other, self._context.scaling_factor)

            if self._context.scaling_factor > 1:
                # Switch self to the smaller context without scaling the
                # plaintext. This divides the plaintext by the scaling factor.
                # The moduli in the new context have one fewer elements than the
                # original context.
                single_scaled_self = self.get_mod_reduced(preserve_plaintext=False)
            else:
                # When the scaling factor is 1, there is no need for any
                # division or mod switching.
                single_scaled_self = self

            if other.shape == []:
                # Multiplying by a scalar uses a special op which is more
                # efficient than the caller creating creating a ShellTensor the
                # same dimensions as self and multiplying.
                if self.is_encrypted:
                    raw_result = shell_ops.mul_ct_tf_scalar64(
                        single_scaled_self._context._raw_context,
                        single_scaled_self,
                        single_scaled_other,
                    )
                else:
                    raw_result = shell_ops.mul_pt_tf_scalar64(
                        single_scaled_self._context._raw_context,
                        single_scaled_self,
                        single_scaled_other,
                    )
                return ShellTensor64(
                    value=raw_result,
                    context=single_scaled_self._context,
                    underlying_dtype=self._underlying_dtype,
                    is_enc=self._is_enc,
                    noise_bit_count=single_scaled_self.noise_bits + other.noise_bits,
                )

            else:
                shell_other = ShellTensor64(
                    value=shell_ops.polynomial_import64(
                        single_scaled_self._context._raw_context, single_scaled_other
                    ),
                    context=single_scaled_self._context,
                    underlying_dtype=self._underlying_dtype,
                    is_enc=False,
                    noise_bit_count=self.noise_bits,
                )
                return ShellTensor64(
                    value=shell_ops.mul_ct_pt64(
                        single_scaled_self._raw, shell_other._raw
                    ),
                    context=single_scaled_self._context,
                    underlying_dtype=self._underlying_dtype,
                    is_enc=self._is_enc or other._is_enc,
                    noise_bit_count=single_scaled_self.noise_bits
                    + self._context.noise_bits,
                )

        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def get_mod_reduced(self, preserve_plaintext=True):
        """Switches the ShellTensor to a new context with different moduli. If
        preserve_plaintext is True (default), the plaintext value will be
        maintained through the modulus switch. If preserve_plaintext is False,
        the plaintext will be divided by the ratio of the new and old moduli."""

        if preserve_plaintext and hasattr(self, "_mod_reduced_pt_preserved"):
            return self._mod_reduced_pt_preserved
        if not preserve_plaintext and hasattr(self, "_mod_reduced_pt_divided"):
            return self._mod_reduced_pt_divided

        # Switch to the new context and moduli.
        if self.is_encrypted:
            raw_result = shell_ops.modulus_reduce_ct64(
                self._context._raw_context,
                self._raw,
                preserve_plaintext,
            )
        else:
            raw_result = shell_ops.modulus_reduce_pt64(
                self._context._raw_context,
                self._raw,
                preserve_plaintext,
            )

        reduced_self = ShellTensor64(
            value=raw_result,
            context=self._context.get_mod_reduced(),
            underlying_dtype=self._underlying_dtype,
            is_enc=self._is_enc,
            noise_bit_count=self.noise_bits
            - self._context.main_moduli[-1].bit_length()
            + 1,
        )

        # Cache the result.
        if preserve_plaintext:
            self._mod_reduced_pt_preserved = reduced_self
        else:
            self._mod_reduced_pt_divided = reduced_self

        return reduced_self

    def roll(self, rotation_key, shift):
        if not isinstance(rotation_key, ShellRotationKey64):
            raise ValueError(
                "Rotation key must be provided. Instead saw {rotation_key}."
            )

        if not self._is_enc:
            raise ValueError("Unencrypted ShellTensor rotation not supported yet.")

        # Get the correct rotation key for the level of this ciphertext.
        raw_rotation_key = rotation_key._get_key_at_level(self._context.level)

        shift = tf.cast(shift, tf.int64)

        return ShellTensor64(
            value=shell_ops.roll64(raw_rotation_key, self._raw, shift),
            context=self._context,
            underlying_dtype=self._underlying_dtype,
            is_enc=True,
            noise_bit_count=self._noise_bit_count + 6,  # TODO correct?
        )

    def reduce_sum(self, axis, rotation_key=None):
        if not self._is_enc:
            raise ValueError("Unencrypted ShellTensor reduce_sum not supported yet.")

        # Check axis is a scalar
        if isinstance(axis, tf.Tensor) and not axis.shape != []:
            raise ValueError("Only scalar `axis` is supported.")

        if axis == 0:
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    "Rotation key must be provided to reduce_sum over the first axis. Instead saw {rotation_key}."
                )

            # Get the correct rotation key for the level of this ciphertext.
            raw_rotation_key = rotation_key._get_key_at_level(self._context.level)

            # reduce sum does log2(num_slots) rotations and additions.
            # TODO: add noise from rotations?
            result_noise_bits = (
                self._noise_bit_count + self._context.num_slots.bit_length() + 1,
            )

            return ShellTensor64(
                value=shell_ops.reduce_sum_by_rotation64(self._raw, raw_rotation_key),
                context=self._context,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                noise_bit_count=result_noise_bits,
            )
        else:
            if axis >= len(self.shape):
                raise ValueError("Axis greater than number of dimensions")

            result_noise_bits = (
                self._noise_bit_count + self.shape[axis].bit_length() + 1
            )

            return ShellTensor64(
                value=shell_ops.reduce_sum64(self._raw, axis),
                context=self._context,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                noise_bit_count=result_noise_bits,
            )


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


def to_shell_tensor(context, tensor):
    if isinstance(tensor, ShellTensor64):
        return tensor
    if isinstance(tensor, tf.Tensor):
        assert isinstance(
            context, ShellContext64
        ), f"Context must be a ShellContext64, instead got {type(context)}"

        if not tensor.dtype.is_floating and context.scaling_factor != 1:
            raise ValueError(
                "Scaling factor only supported for floating point datatypes."
            )

        # Shell tensor keeps all plaintexts at scaling_factor**2, so multiply by
        # scaling_factor**2 to get the plaintext that shell should see.
        scaled_tensor = _encode_scaling(
            tensor, context.scaling_factor * context.scaling_factor
        )

        return ShellTensor64(
            value=shell_ops.polynomial_import64(context._raw_context, scaled_tensor),
            context=context,
            underlying_dtype=tensor.dtype,
            is_enc=False,
            noise_bit_count=context.noise_bits,
        )
    else:
        raise ValueError("Cannot convert to ShellTensor64")


def from_shell_tensor(s_tensor):
    assert isinstance(
        s_tensor, ShellTensor64
    ), f"Should be ShellTensor, instead got {type(s_tensor)}"

    if s_tensor.is_encrypted:
        # return NotImplemented  # may allow other conversions
        raise ValueError("Cannot convert encrypted ShellTensor to tf. Decrypt first.")

    shell_export_type = _get_shell_dtype_from_underlying(s_tensor._underlying_dtype)

    # Convert from polynomial representation to plaintext tensorflow tensor.
    # Always convert to int64, then handle the fixed point as appropriate.
    tf_tensor = shell_ops.polynomial_export64(
        s_tensor._context._raw_context,
        s_tensor._raw,
        dtype=shell_export_type,
    )

    # Shell tensor keeps all plaintexts at scaling_factor**2, so divide by
    # scaling_factor**2 to get the original plaintext.
    return _decode_scaling(
        tf_tensor,
        s_tensor._underlying_dtype,
        s_tensor._context.scaling_factor * s_tensor._context.scaling_factor,
    )


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

        # Convert y to 1x the scaling factor.
        single_scaled_y = _encode_scaling(y, x._context.scaling_factor)

        # Switch x to a smaller context without scaling the plaintext. This
        # divides the plaintext by the scaling factor. The moduli in the new
        # context have one fewer elements than the original context.
        single_scaled_x = x.get_mod_reduced(preserve_plaintext=False)

        # Noise grows from one multiplication then a sum over that dimension.
        multiplication_noise = x.noise_bits + x._context.noise_bits
        reduce_sum_noise = multiplication_noise + x.shape[1].bit_length()

        return ShellTensor64(
            value=shell_ops.mat_mul_ct_pt64(
                single_scaled_x._context._raw_context,
                single_scaled_x._raw,
                single_scaled_y,
            ),
            context=single_scaled_x._context,
            underlying_dtype=x._underlying_dtype,
            is_enc=True,
            noise_bit_count=reduce_sum_noise,
        )

    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
        if not isinstance(rotation_key, ShellRotationKey64):
            return ValueError(
                "Rotation key must be provided to matmul pt*ct. Instead saw {rotation_key}."
            )

        if x.dtype != y._underlying_dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x.dtype} and {y._underlying_dtype}"
            )

        # Convert x to 1x the scaling factor.
        single_scaled_x = _encode_scaling(x, y._context.scaling_factor)

        # Switch y to a smaller context without scaling the plaintext. This
        # divides the plaintext by the scaling factor. The moduli in the new
        # context have one fewer elements than the original context.
        single_scaled_y = y.get_mod_reduced(preserve_plaintext=False)

        # Get the correct rotation key for the level of y.
        raw_rotation_key = rotation_key._get_key_at_level(
            single_scaled_y._context.level
        )

        # Noise grows from doing one multiplication then a reduce_sum operation
        # over the outer (ciphertext) dimension. dimension. The noise from the
        # reduce_sum is a rough estimate that works for slots = 2**11.
        multiplication_noise = y._noise_bit_count + 1
        rotation_noise = multiplication_noise + 60
        reduce_sum_noise = rotation_noise + y._context.num_slots.bit_length()

        return ShellTensor64(
            value=shell_ops.mat_mul_pt_ct64(
                single_scaled_y._context._raw_context,
                raw_rotation_key,
                single_scaled_x,
                single_scaled_y._raw,
            ),
            context=single_scaled_y._context,
            underlying_dtype=y._underlying_dtype,
            is_enc=True,
            noise_bit_count=reduce_sum_noise,
        )

    elif isinstance(x, ShellTensor64) and isinstance(y, ShellTensor64):
        return NotImplemented

    elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        return tf.matmul(x, y)

    else:
        raise ValueError(
            f"Unsupported types for matmul. Got {type(x)} and {type(y)}. If multiplying by a plaintext, pass it as a plain TensorFlow tensor, not a ShellTensor."
        )
