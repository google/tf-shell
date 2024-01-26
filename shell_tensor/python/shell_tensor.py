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
import shell_tensor.python.ops.shell_ops as shell_ops
import shell_tensor.shell as raw_bindings


class ShellTensor64(object):
    is_tensor_like = True  # needed to pass tf.is_tensor, new as of TF 2.2+

    def __init__(
        self,
        value,
        context,
        num_slots,
        underlying_dtype,
        is_enc=False,
        fxp_fractional_bits=0,
        mult_count=0,
    ):
        assert isinstance(
            value, tf.Tensor
        ), f"Should be variant tensor, instead got {type(value)}"
        assert (
            value.dtype is tf.variant
        ), f"Should be variant tensor, instead got {value.dtype}"
        self._raw = value
        self._context = context
        self._num_slots = num_slots
        self._underlying_dtype = underlying_dtype
        self._is_enc = is_enc

        # Fixed point parameters.
        # ShellTensor operates on fixed point numbers. It will automatically
        # convert into and out of fixed point representation before wrapping in
        # shell datatypes (both plaintext and ciphertext). When doing fixed
        # point multiplication, the number of fractional bits doubles with
        # multiplication. Usually this would be immediately followed by a right
        # shift to bring the precision back down to the number of fractional
        # bits. Right shifting encrypted datatypes is very expensive. Instead
        # ShellTensor keeps track of the number of multiplications that have
        # occured, then right shift for all multiplications together at the end.
        # This requires keeping track of the number of multiplications that have
        # occured. This is the mult_count parameter.
        #
        # Note that now adding/subtracting/multiplying two ShellTensors together
        # is more complicated as each could have a different number of
        # multiplications, and thus a different number of fractional bits. The
        # operand with fewer mult_count must be scaled up by
        # 2**(difference_in_mult_count * fractional_bits) to match the other
        # operand. _self_at_fxp_multiplier caches scaled up versions of itself
        # for each previously requested mult_count.
        self._fxp_fractional_bits = fxp_fractional_bits
        self._mult_count = (
            mult_count  # number of preceeding multiplications resulting in self.
        )
        self._self_at_fxp_multiplier = {mult_count: self}

    @property
    def shape(self):
        return self._num_slots + self._raw.shape

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

    def num_fxp_fractional_bits(self):
        return self._fxp_fractional_bits * (2**self._mult_count)

    def get_encrypted(self, key):
        if self._is_enc:
            return self
        else:
            return ShellTensor64(
                value=shell_ops.encrypt64(self._context, key, self._raw),
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=self._mult_count,
            )

    def get_decrypted(self, key=None):
        if not self._is_enc:
            return from_shell_tensor(self)
        else:
            if key is None:
                raise ValueError(
                    "Key must be provided to decrypt encrypted ShellTensor."
                )
            # Find out what dtype shell thinks the plaintext is.
            fxp_dtype = _get_fxp_dtype_from_underlying(self._underlying_dtype)

            # Decrypt op returns a tf Tensor.
            tf_tensor = shell_ops.decrypt64(
                self._context,
                key,
                self._raw,
                dtype=fxp_dtype,
            )

            # Convert out of fixed point to the underlying dtype.
            return _from_fixed_point(
                tf_tensor,
                self.num_fxp_fractional_bits(),
                self._underlying_dtype,
            )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            max_mult_count = max(self._mult_count, other._mult_count)
            matched_other = other.get_at_multiplication_count(max_mult_count)
            matched_self = self.get_at_multiplication_count(max_mult_count)

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
                    self._context, matched_self._raw, matched_other._raw
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=result_raw,
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc or other._is_enc,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=max_mult_count,
            )

        elif isinstance(other, tf.Tensor):
            # TODO(jchoncholas): Adding a scalar uses a special op that is
            # more efficient.
            if other.shape == []:
                raise ValueError("Scalar addition not yet implemented.")

            # Lift tensorflow tensor to shell tensor with the same number
            # fractional bits as self.
            so = to_shell_tensor(self._context, other, self._fxp_fractional_bits)
            return self + so

        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ShellTensor64):
            max_mult_count = max(self._mult_count, other._mult_count)
            matched_other = other.get_at_multiplication_count(max_mult_count)
            matched_self = self.get_at_multiplication_count(max_mult_count)

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
                    self._context, matched_self._raw, matched_other._raw
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=result_raw,
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc or other._is_enc,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=max_mult_count,
            )
        elif isinstance(other, tf.Tensor):
            # TODO(jchoncholas): Subtracting a scalar uses a special op that is
            # more efficient.
            if other.shape == []:
                raise ValueError("Scalar subtraction not yet implemented.")

            # Lift tensorflow tensor to shell tensor with the same number
            # fractional bits as self.
            so = to_shell_tensor(self._context, other, self._fxp_fractional_bits)
            return self - so
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, tf.Tensor):
            shell_other = to_shell_tensor(
                self._context, other, self._fxp_fractional_bits
            )
            matched_shell_other = shell_other.get_at_multiplication_count(
                self._mult_count
            )

            if self.is_encrypted:
                negative_self = -self
                raw_result = shell_ops.add_ct_pt64(
                    negative_self._raw, matched_shell_other._raw
                )
            else:
                raw_result = shell_ops.sub_pt_pt64(
                    self._context, matched_shell_other._raw, self._raw
                )

            return ShellTensor64(
                value=raw_result,
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=self._mult_count,
            )
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_encrypted:
            raw_result = shell_ops.neg_ct64(self._raw)
        else:
            raw_result = shell_ops.neg_pt64(self._context, self._raw)

        return ShellTensor64(
            value=raw_result,
            context=self._context,
            num_slots=self._num_slots,
            underlying_dtype=self._underlying_dtype,
            is_enc=self._is_enc,
            fxp_fractional_bits=self._fxp_fractional_bits,
            mult_count=self._mult_count,
        )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            max_mult_count = max(self._mult_count, other._mult_count)
            matched_other = other.get_at_multiplication_count(max_mult_count)
            matched_self = self.get_at_multiplication_count(max_mult_count)

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
                    self._context, matched_self._raw, matched_other._raw
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                value=raw_result,
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc or other._is_enc,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=max_mult_count + 1,
            )
        elif isinstance(other, tf.Tensor):
            # Multiplying by a scalar uses a special op which is more efficient
            # than the caller creating creating a ShellTensor the same
            # dimensions as self and multiplying.
            if other.shape == []:
                # Convert other to fixed point. Using num_fxp_fractional_bits()
                # ensure's it has the same number of fractional bits as self,
                # taking multiplicative depth into account.
                fxp_tensor = _to_fixed_point(other, self.num_fxp_fractional_bits())

                if self.is_encrypted:
                    raw_result = shell_ops.mul_ct_tf_scalar64(
                        self._context, self._raw, fxp_tensor
                    )
                else:
                    raw_result = shell_ops.mul_pt_tf_scalar64(
                        self._context, self._raw, fxp_tensor
                    )
                return ShellTensor64(
                    value=raw_result,
                    context=self._context,
                    num_slots=self._num_slots,
                    underlying_dtype=self._underlying_dtype,
                    is_enc=self._is_enc,
                    fxp_fractional_bits=self._fxp_fractional_bits,
                    mult_count=self._mult_count + 1,
                )

            # Lift tensorflow tensor to shell tensor before multiplication.
            so = to_shell_tensor(self._context, other, self._fxp_fractional_bits)
            return self * so

        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def roll(self, rotation_key, num_slots):
        if not self._is_enc:
            raise ValueError("Unencrypted ShellTensor rotation not supported yet.")
        else:
            return ShellTensor64(
                value=shell_ops.roll64(rotation_key, self._raw, num_slots),
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=self._mult_count,
            )

    def reduce_sum(self, rotation_key):
        if not self._is_enc:
            raise ValueError("Unencrypted ShellTensor reduce_sum not supported yet.")
        else:
            return ShellTensor64(
                value=shell_ops.reduce_sum64(rotation_key, self._raw),
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=True,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=self._mult_count,
            )

    def get_at_multiplication_count(self, mult_count):
        """Returns a ShellTensor whose values have been left shifted to match
        the specified multiplicative depth. Fixed point multiplication doubles
        the number of fractional bits. Since ShellTensor does not right shift
        after multiplication, (only after decryption) two ShellTensors may have
        different number of fractional bits. This function will return a new
        ShellTensor with the same value as itself but left shifted to match the
        specified multiplicative depth.

        This function will cache the result for future calls with the same
        multiplicative depth.

        For now, this function only supports increasing the number of
        multiplications. Decreasing the number of multiplications is possible
        and would require right shifting by 2^(-x) mod t, a multiplicative
        inverse in the plaintext modulus's field.
        """
        if self._mult_count > mult_count:
            raise ValueError("Cannot reduce multiplication count of ShellTensor64.")
        elif mult_count in self._self_at_fxp_multiplier:
            return self._self_at_fxp_multiplier[mult_count]
        else:
            num_mul_to_do = mult_count - self._mult_count

            wanted_fxp_fractional_bits = self._fxp_fractional_bits * (
                2**num_mul_to_do
            )
            needed_fxp_fractional_bits = (
                wanted_fxp_fractional_bits - self._fxp_fractional_bits
            )
            fxp_multiplier = tf.constant(
                2**needed_fxp_fractional_bits, dtype=tf.int64
            )

            # Perform the multiplication.
            if self.is_encrypted:
                raw_result = shell_ops.mul_ct_tf_scalar64(
                    self._context, self._raw, fxp_multiplier
                )
            else:
                raw_result = shell_ops.mul_pt_tf_scalar64(
                    self._context, self._raw, fxp_multiplier
                )
            left_shifted = ShellTensor64(
                value=raw_result,
                context=self._context,
                num_slots=self._num_slots,
                underlying_dtype=self._underlying_dtype,
                is_enc=self._is_enc,
                fxp_fractional_bits=self._fxp_fractional_bits,
                mult_count=mult_count,  # Override the mult_count, may be higher
                # than self._mult_count + 1 if mult_count_to_do was larger than
                # 1.
            )
            self._self_at_fxp_multiplier[mult_count] = left_shifted
            return left_shifted


# This class uses a fixed point dtype depending on the dtype of tensorflow
# tensor. The fixed point dtype is what is stored in shell plain texts and
# ciphertexts.
def _get_fxp_dtype_from_underlying(type):
    if type in [tf.float32, tf.float64]:
        return tf.int64
    elif type in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        return tf.uint64
    elif type in [tf.int8, tf.int16, tf.int32, tf.int64]:
        return tf.int64
    else:
        raise ValueError(f"Unsupported type {type}")


def _to_fixed_point(tf_tensor, fxp_fractional_bits):
    if tf_tensor.dtype in [tf.float32, tf.float64]:
        fxp_fractional_multiplier = 2**fxp_fractional_bits
        integer = tf.cast(tf_tensor, tf.int64) * fxp_fractional_multiplier
        fractional = tf.cast(tf.math.mod(tf_tensor, 1), tf.int64)
        return integer + fractional
    elif tf_tensor.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        # If the tensor dtype is uint64, we assume its fixed point representation
        # also fits in a uint64.
        fxp_tensor = tf.cast(tf_tensor, tf.uint64)
        return tf.bitwise.left_shift(fxp_tensor, fxp_fractional_bits)
    elif tf_tensor.dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
        # If the tensor dtype is int64, we assume it fits in an int64 in the
        # fixed point representation (after left shifting).
        fxp_tensor = tf.cast(tf_tensor, tf.int64)
        return tf.bitwise.left_shift(fxp_tensor, fxp_fractional_bits)
    else:
        raise ValueError(f"Unsupported dtype {tf_tensor.dtype}")


def _from_fixed_point(fxp_tensor, fxp_fractional_bits, output_dtype):
    if output_dtype in [tf.float32, tf.float64]:
        assert fxp_tensor.dtype == tf.int64
        fxp_fractional_multiplier = 2**fxp_fractional_bits
        integer = tf.cast(
            tf.bitwise.right_shift(fxp_tensor, fxp_fractional_bits), output_dtype
        )
        fractional_mask = (
            tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64), fxp_fractional_bits)
            - 1
        )
        fractional = tf.cast(
            tf.bitwise.bitwise_and(fxp_tensor, fractional_mask), output_dtype
        )
        return integer + fractional / fxp_fractional_multiplier
    elif output_dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
        # When returning an integer datatype, the fractional bits of the fixed
        # point representation cannot be stored. Throw the low precision bits
        # away. TODO(jchoncholas): Round
        assert fxp_tensor.dtype == tf.uint64
        tf_tensor = tf.bitwise.right_shift(fxp_tensor, fxp_fractional_bits)
        # round_up = tf.bitwise.right_shift(fxp_tensor, fxp_fractional_bits - 1)
        # round_up = tf.bitwise.bitwise_and(round_up, 1)
        # tf_tensor += round_up
        return tf.cast(tf_tensor, output_dtype)
    elif output_dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
        assert fxp_tensor.dtype == tf.int64
        tf_tensor = tf.bitwise.right_shift(fxp_tensor, fxp_fractional_bits)
        # round_up = tf.bitwise.right_shift(fxp_tensor, fxp_fractional_bits - 1)
        # round_up = tf.bitwise.bitwise_and(round_up, 1)
        # tf_tensor += round_up
        return tf.cast(tf_tensor, output_dtype)
    else:
        raise ValueError(f"Unsupported dtype {output_dtype}")


def to_shell_tensor(context, tensor, fxp_fractional_bits=0):
    if isinstance(tensor, ShellTensor64):
        return tensor
    if isinstance(tensor, tf.Tensor):
        # Convert to fixed point.
        fxp_tensor = _to_fixed_point(tensor, fxp_fractional_bits)

        return ShellTensor64(
            value=shell_ops.polynomial_import64(context, fxp_tensor),
            context=context,
            num_slots=tensor.shape[0],
            underlying_dtype=tensor.dtype,
            fxp_fractional_bits=fxp_fractional_bits,
            mult_count=0,
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

    shell_export_type = _get_fxp_dtype_from_underlying(s_tensor._underlying_dtype)

    # Convert from polynomial representation to plaintext tensorflow tensor.
    # Always convert to int64, then handle the fixed point as appropriate.
    tf_tensor = shell_ops.polynomial_export64(
        s_tensor._context,
        s_tensor._raw,
        dtype=shell_export_type,
    )

    return _from_fixed_point(
        tf_tensor,
        s_tensor.num_fxp_fractional_bits(),
        s_tensor._underlying_dtype,
    )


def create_context64(
    log_n, main_moduli, aux_moduli, plaintext_modulus, noise_variance, seed=""
):
    return shell_ops.context_import64(
        log_n=log_n,
        main_moduli=main_moduli,
        aux_moduli=aux_moduli,
        plaintext_modulus=plaintext_modulus,
        noise_variance=noise_variance,
        seed=seed,
    )


def create_key64(context):
    return shell_ops.key_gen64(context)


def create_rotation_key64(context, key):
    return shell_ops.rotation_key_gen64(context, key)


def matmul(x, y, rotation_key=None):
    if isinstance(x, ShellTensor64) and isinstance(y, tf.Tensor):
        # Convert y to fixed point and make sure it's multiplication level
        # matches x's.
        fxp_tensor = _to_fixed_point(y, x.num_fxp_fractional_bits())

        return ShellTensor64(
            value=shell_ops.mat_mul_ct_pt64(x._context, x._raw, fxp_tensor),
            context=x._context,
            num_slots=x._num_slots,
            underlying_dtype=x._underlying_dtype,
            is_enc=True,
            fxp_fractional_bits=x._fxp_fractional_bits,
            mult_count=x._mult_count + 1,
        )

    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
        if rotation_key is None:
            return ValueError("Rotation key must be provided to multiply pt*ct")

        if x.dtype != y._underlying_dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x.dtype} and {y._underlying_dtype}"
            )

        # Convert x to fixed point and make sure it's multiplication level
        # matches y's.
        fxp_tensor = _to_fixed_point(x, y.num_fxp_fractional_bits())

        return ShellTensor64(
            value=shell_ops.mat_mul_pt_ct64(
                y._context, rotation_key, fxp_tensor, y._raw
            ),
            context=y._context,
            num_slots=y._num_slots,
            underlying_dtype=y._underlying_dtype,
            is_enc=True,
            fxp_fractional_bits=y._fxp_fractional_bits,
            mult_count=y._mult_count + 1,
        )

    elif isinstance(x, ShellTensor64) and isinstance(y, ShellTensor64):
        return NotImplemented

    elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        return tf.matmul(x, y)

    else:
        raise ValueError(
            f"Unsupported types for matmul. Got {type(x)} and {type(y)}. If multiplying a plaintext, pass it as a plain TensorFlow tensor, not a ShellTensor."
        )
