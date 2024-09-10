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
import tf_shell.python.shell_ops as shell_ops
from tf_shell.python.shell_context import ShellContext64
from tf_shell.python.shell_context import mod_reduce_context64
from tf_shell.python.shell_key import ShellKey64
from tf_shell.python.shell_key import mod_reduce_key64
from tf_shell.python.shell_key import ShellRotationKey64
from tf_shell.python.shell_key import ShellFastRotationKey64


class ShellTensor64(tf.experimental.ExtensionType):
    _raw_tensor: tf.Tensor
    _context: ShellContext64
    _underlying_dtype: tf.DType
    _scaling_factor: int
    _is_enc: bool
    _is_fast_rotated: bool = False

    @property
    def shape(self):
        try:
            return tf.TensorShape([self._context.num_slots.numpy()]).concatenate(
                self._raw_tensor.get_shape()
            )
        except AttributeError:
            return tf.TensorShape([None]).concatenate(self._raw_tensor.get_shape())

    @property
    def ndim(self):
        return self._raw_tensor.ndim + 1

    @property
    def dtype(self):
        return tf.variant

    @property
    def name(self):
        return self._raw_tensor.name

    @property
    def plaintext_dtype(self):
        return self._underlying_dtype

    @property
    def is_encrypted(self):
        return self._is_enc

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
            _raw_tensor=self._raw_tensor[slice[1:]],
            _context=self._context,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self.is_encrypted,
        )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_ct64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._raw_context,
                    matched_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.add_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=result_raw_tensor,
                _context=matched_self._context,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=self._scaling_factor,
                _is_enc=self._is_enc or other._is_enc,
            )

        elif isinstance(other, tf.Tensor):
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar addition, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other, tf.expand_dims(self._context.num_slots, 0)
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                # In the special case of broadcasting over the packing'
                # dimension, replicate the scalar across all slots.
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

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
                result_raw_tensor = shell_ops.sub_ct_ct64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.sub_ct_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                negative_other = -matched_other
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._raw_context,
                    negative_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.sub_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=result_raw_tensor,
                _context=matched_self._context,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=self._scaling_factor,
                _is_enc=self._is_enc or other._is_enc,
            )
        elif isinstance(other, tf.Tensor):
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other, tf.expand_dims(self._context.num_slots, 0)
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

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
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

            # Import to a shell plaintext, which pads the first dimension with
            # zeros out to the number of slots.
            shell_other = to_shell_plaintext(other, self._context)

            if self.is_encrypted:
                negative_self = -self
                raw_result = shell_ops.add_ct_pt64(
                    self._context._raw_context,
                    negative_self._raw_tensor,
                    shell_other._raw_tensor,
                )
            else:
                raw_result = shell_ops.sub_pt_pt64(
                    self._context._raw_context,
                    shell_other._raw_tensor,
                    self._raw_tensor,
                )

            return ShellTensor64(
                _raw_tensor=raw_result,
                _context=self._context,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=self._scaling_factor,
                _is_enc=self._is_enc,
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
            raw_result = shell_ops.neg_ct64(
                self._context._raw_context, self._raw_tensor
            )
        else:
            raw_result = shell_ops.neg_pt64(
                self._context._raw_context, self._raw_tensor
            )

        return ShellTensor64(
            _raw_tensor=raw_result,
            _context=self._context,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self._is_enc,
        )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                if self._is_fast_rotated or other._is_fast_rotated:
                    raise ValueError(
                        "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be multiplied with another ciphertext."
                    )
                raw_result = shell_ops.mul_ct_ct64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_self._context._raw_context,
                    matched_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_pt_pt64(
                    matched_self._context._raw_context,
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=raw_result,
                _context=matched_self._context,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=matched_self._scaling_factor**2,
                _is_enc=self._is_enc or other._is_enc,
            )
        elif isinstance(other, tf.Tensor):
            # Multiplying by a scalar uses a special op which is more efficient
            # than the caller creating a ShellTensor the same dimensions as self
            # and multiplying. Note this includes the case when broadcasting
            # over the packing dimension.
            if other.shape == () or other.shape[0] == 1:
                # If not a scalar, remove the outer dim (which is always 1).
                if other.shape != ():
                    assert other.shape[0] == 1
                    other = tf.reshape(other, other.shape[1:])

                # Encode the other scalar tensor to the same scaling factor as
                # self.
                other = _encode_scaling(other, self._scaling_factor)

                if self.is_encrypted:
                    raw_result = shell_ops.mul_ct_tf_scalar64(
                        self._context._raw_context, self._raw_tensor, other
                    )
                else:
                    raw_result = shell_ops.mul_pt_tf_scalar64(
                        self._context._raw_context, self._raw_tensor, other
                    )

                return ShellTensor64(
                    _raw_tensor=raw_result,
                    _context=self._context,
                    _underlying_dtype=self._underlying_dtype,
                    _scaling_factor=self._scaling_factor**2,
                    _is_enc=self._is_enc,
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


def mod_reduce_tensor64(shell_tensor):
    """Switches the ShellTensor to a new context with different moduli. If
    preserve_plaintext is True (default), the plaintext value will be
    maintained through the modulus switch. If preserve_plaintext is False,
    the plaintext will be divided by the ratio of the new and old moduli."""

    assert isinstance(
        shell_tensor, ShellTensor64
    ), f"shell_tensor must be a ShellTensor64, instead got {type(shell_tensor)}"

    # Switch to the new context and moduli.
    if shell_tensor.is_encrypted:
        op = shell_ops.modulus_reduce_ct64
    else:
        op = shell_ops.modulus_reduce_pt64

    raw_result = op(
        shell_tensor._context._raw_context,
        shell_tensor._raw_tensor,
    )

    reduced_self = ShellTensor64(
        _raw_tensor=raw_result,
        _context=mod_reduce_context64(shell_tensor._context),
        _underlying_dtype=shell_tensor._underlying_dtype,
        _scaling_factor=shell_tensor._scaling_factor,
        _is_enc=shell_tensor._is_enc,
    )

    return reduced_self


def _match_moduli_and_scaling(x, y):
    # Mod switch to the smaller modulus of the two.
    while x._context.level > y._context.level:
        x = mod_reduce_tensor64(x)
    while x._context.level < y._context.level:
        y = mod_reduce_tensor64(y)

    # Match the scaling factors.
    # First make sure the scaling factors are compatible.
    frac = x._scaling_factor / y._scaling_factor
    if abs(frac - int(frac)) != 0:
        raise ValueError(
            f"Scaling factors must be compatible. Got {x._scaling_factor} and {y._scaling_factor}"
        )

    while x._scaling_factor > y._scaling_factor:
        y = y.__mul__(x._scaling_factor)
    while x._scaling_factor < y._scaling_factor:
        x = x.__mul__(y._scaling_factor)

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
        if tf.executing_eagerly():
            # In eager mode, we know the number of slots at graph construction
            # time and can check the tensor is the correct size.
            if scaled_tensor.shape[0] > context.num_slots:
                raise ValueError(
                    f"Tensor first dimension is too large. Maximum is {context.num_slots}, got {scaled_tensor.shape[0]}."
                )
            elif scaled_tensor.shape[0] < context.num_slots:
                padding = [[0, context.num_slots - scaled_tensor.shape[0]]] + [
                    [0, 0] for _ in range(len(scaled_tensor.shape) - 1)
                ]
                scaled_tensor = tf.pad(scaled_tensor, padding)
        else:
            # In graph mode, we may not know the number of slots until runtime.
            # Try the padding, but if it fails (e.g. the batching dimension is
            # too large), the user will see the error when the tensor is used in
            # a SHELL operation at runtime.
            try:
                padding = [[0, context.num_slots - scaled_tensor.shape[0]]] + [
                    [0, 0] for _ in range(len(scaled_tensor.shape) - 1)
                ]
                scaled_tensor = tf.pad(scaled_tensor, padding)
            except:
                pass

        return ShellTensor64(
            _raw_tensor=shell_ops.polynomial_import64(
                context._raw_context, scaled_tensor
            ),
            _context=context,
            _underlying_dtype=tensor.dtype,
            _scaling_factor=context.scaling_factor,
            _is_enc=False,
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
                _raw_tensor=shell_ops.encrypt64(
                    x._context._raw_context,
                    key._raw_key,
                    x._raw_tensor,
                ),
                _context=x._context,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
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

    try:
        batching_dim = s_tensor._context.num_slots.numpy()
    except AttributeError:
        batching_dim = -1

    if s_tensor.is_encrypted and s_tensor._is_fast_rotated:
        if not isinstance(key, ShellFastRotationKey64):
            raise ValueError(
                "ShellFastRotationKey must be provided to decrypt a fast-rotated ShellTensor."
            )

        # Get the correct rotation key for the level of this ciphertext.
        raw_rotation_key = key._get_key_at_level(s_tensor._context.level)

        # Decrypt op returns a tf Tensor.
        tf_tensor = shell_ops.decrypt_fast_rotated64(
            context=s_tensor._context._raw_context,
            fast_rotation_key=raw_rotation_key,
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
        )

    elif s_tensor.is_encrypted:
        if not isinstance(key, ShellKey64):
            raise ValueError(
                "Key must be provided to decrypt an encrypted ShellTensor."
            )

        # Mod reduce the key to match the level of the ciphertext.
        while key.level > s_tensor._context.level:
            key = mod_reduce_key64(key)

        # Decrypt op returns a tf Tensor.
        tf_tensor = shell_ops.decrypt64(
            context=s_tensor._context._raw_context,
            key=key._raw_key,
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
        )

    elif not s_tensor.is_encrypted:
        # Convert from polynomial representation to plaintext tensorflow tensor.
        # Always convert to int64, then handle the fixed point as appropriate.
        tf_tensor = shell_ops.polynomial_export64(
            shell_context=s_tensor._context._raw_context,
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
        )

    else:
        raise ValueError(f"Invalid ShellTensor state. Got {s_tensor}.")

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
            _raw_tensor=shell_ops.roll64(raw_rotation_key, x._raw_tensor, shift),
            _context=x._context,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=True,
        )
    elif isinstance(x, tf.Tensor):
        return tf.roll(x, shift)
    else:
        raise ValueError(f"Unsupported type for roll. Got {type(x)}.")


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

            return ShellTensor64(
                _raw_tensor=shell_ops.reduce_sum_by_rotation64(
                    raw_rotation_key, x._raw_tensor
                ),
                _context=x._context,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
            )

        else:
            return ShellTensor64(
                _raw_tensor=shell_ops.reduce_sum64(x._raw_tensor, axis=axis),
                _context=x._context,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
            )
    elif isinstance(x, tf.Tensor):
        return tf.reduce_sum(x, axis)
    else:
        raise ValueError(f"Unsupported type for reduce_sum. Got {type(x)}.")


def fast_reduce_sum(x):
    """Fast reduce sum is a special case of reduce sum where the encrypted
    input is reduce_summed, however it skips the keyswitching so the resulting
    ciphertext is no longer valid under the original secret key. The plaintext
    can still be recovered, through a special decryption process. This means
    a fast_reduce_summed ciphertext has limited subsequent operations, i.e. only
    add / multiply by plaintexts are supported. See the op kernel for a more
    technical explanation."""
    if not isinstance(x, ShellTensor64):
        raise ValueError("Input must be ShellTensor.")
    if not x._is_enc:
        raise ValueError("Unencrypted fast_reduce_sum not supported yet.")

    return ShellTensor64(
        _raw_tensor=shell_ops.fast_reduce_sum_by_rotation64(
            x._context._raw_context, x._raw_tensor
        ),
        _context=x._context,
        _underlying_dtype=x._underlying_dtype,
        _scaling_factor=x._scaling_factor,
        _is_enc=True,
        _is_fast_rotated=True,
    )


def matmul(x, y, rotation_key=None, fast=False):
    """Matrix multiplication is specialized to whether the operands are
    plaintext or ciphertext.

    matmul(ciphertext, plaintext) works the same way as Tensorflow.

    matmul(plaintext, ciphertext) in tf-shell has slightly different semantics
    than plaintext / Tensorflow. tf-shell affects top and bottom halves
    independently, as well as the first dimension repeating the sum of either
    the halves."""

    if len(x.shape) < 2 or len(y.shape) < 2:
        raise ValueError(
            f"matmul not supported for tensors with rank < 2. Got {x.shape} and {y.shape}."
        )

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

        return ShellTensor64(
            _raw_tensor=shell_ops.mat_mul_ct_pt64(
                x._context._raw_context,
                x._raw_tensor,
                scaled_y,
            ),
            _context=x._context,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor**2,
            _is_enc=True,
        )

    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
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

        if fast:
            return ShellTensor64(
                _raw_tensor=shell_ops.fast_mat_mul_pt_ct64(
                    y._context._raw_context,
                    scaled_x,
                    y._raw_tensor,
                    # no rotation key
                ),
                _context=y._context,
                _underlying_dtype=y._underlying_dtype,
                _scaling_factor=y._scaling_factor**2,
                _is_enc=True,
                _is_fast_rotated=True,
            )
        else:
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    f"Rotation key must be provided to matmul pt*ct. Instead saw {rotation_key}."
                )

            # Get the correct rotation key for the level of y.
            raw_rotation_key = rotation_key._get_key_at_level(y._context.level)

            return ShellTensor64(
                _raw_tensor=shell_ops.mat_mul_pt_ct64(
                    y._context._raw_context,
                    scaled_x,
                    y._raw_tensor,
                    raw_rotation_key,
                ),
                _context=y._context,
                _underlying_dtype=y._underlying_dtype,
                _scaling_factor=y._scaling_factor**2,
                _is_enc=True,
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
            _raw_tensor=shell_ops.expand_dims_variant(x._raw_tensor, axis=axis),
            _context=x._context,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
        )
    elif isinstance(x, tf.Tensor):
        return tf.expand_dims(x, axis)
    else:
        raise ValueError("Unsupported type for expand_dims")


def reshape(x, shape):
    if isinstance(x, ShellTensor64):
        # Perform some checks on the new shape.
        if tf.executing_eagerly() and shape[0] != x._context.num_slots:
            raise ValueError(
                "Cannot reshape axis 0 for ShellTensor64, this is the batching dimension."
            )
        return ShellTensor64(
            _raw_tensor=tf.reshape(x._raw_tensor, shape[1:]),
            _context=x._context,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
        )
    elif isinstance(x, tf.Tensor):
        return tf.reshape(x, shape)
    else:
        raise ValueError("Unsupported type for expand_dims")


def broadcast_to(x, shape):
    if isinstance(x, ShellTensor64):
        if tf.executing_eagerly() and shape[0] != x._context.num_slots:
            raise ValueError(
                "Cannot broadcast_to over axis 0 for ShellTensor64, this is the batching dimension."
            )

        return ShellTensor64(
            _raw_tensor=tf.broadcast_to(x._raw_tensor, shape[1:]),
            _context=x._context,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
        )
    elif isinstance(x, tf.Tensor):
        return tf.broadcast_to(x, shape)
    else:
        raise ValueError("Unsupported type for expand_dims")


def segment_sum(x, segments, num_segments, rotation_key=None):
    if not isinstance(segments, tf.Tensor):
        raise ValueError("`segments` must be a TensorFlow tensor.")

    if isinstance(x, ShellTensor64):
        if not isinstance(rotation_key, ShellRotationKey64):
            raise ValueError(
                f"Rotation key must be provided. Instead saw {rotation_key}."
            )
        raw_rotation_key = rotation_key._get_key_at_level(x._context.level)

        raw_result, reduction_count = shell_ops.segment_sum_ct(
            x._context._raw_context,
            x._raw_tensor,
            segments,
            num_segments,
            raw_rotation_key,
        )

        return (
            ShellTensor64(
                _raw_tensor=raw_result,
                _context=x._context,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=x._is_enc,
            ),
            reduction_count,
        )
    elif isinstance(x, tf.Tensor):
        return tf.math.unsorted_segment_sum(x, segments, num_segments)
    else:
        raise ValueError("Unsupported type for segment_sum")
