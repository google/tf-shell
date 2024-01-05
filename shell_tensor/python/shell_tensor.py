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

    def __init__(self, value, context, num_slots, underlying_dtype, is_enc=False):
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

    def get_encrypted(self, key):
        if self._is_enc:
            return self
        else:
            return ShellTensor64(
                shell_ops.encrypt64(self._context, key, self._raw),
                self._context,
                self._num_slots,
                self._underlying_dtype,
                is_enc=True,
            )

    def get_decrypted(self, key):
        if not self._is_enc:
            return self
        else:
            # Decrypt op returns a tf Tensor
            return shell_ops.decrypt64(
                self._context,
                key,
                self._raw,
                dtype=self._underlying_dtype,
            )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            if self.is_encrypted and other.is_encrypted:
                return ShellTensor64(
                    shell_ops.add_ct_ct64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.add_ct_pt64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and other.is_encrypted:
                return ShellTensor64(
                    shell_ops.add_ct_pt64(other._raw, self._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.add_pt_pt64(self._context, self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    False,
                )
            else:
                raise ValueError("Invalid operands")
        elif isinstance(other, tf.Tensor):
            # lift tensorflow tensor to shell tensor before add
            so = to_shell_tensor(self._context, other)
            return self + so

        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ShellTensor64):
            if self.is_encrypted and other.is_encrypted:
                return ShellTensor64(
                    shell_ops.sub_ct_ct64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.sub_ct_pt64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and other.is_encrypted:
                negative_other = -other
                return ShellTensor64(
                    shell_ops.add_ct_pt64(negative_other._raw, self._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.sub_pt_pt64(self._context, self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    False,
                )
            else:
                raise ValueError("Invalid operands")
        elif isinstance(other, tf.Tensor):
            # lift tensorflow tensor to shell tensor before sub
            so = to_shell_tensor(self._context, other)
            return self - so
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, tf.Tensor):
            so = to_shell_tensor(self._context, other)
            if self.is_encrypted:
                negative_self = -self
                return ShellTensor64(
                    shell_ops.add_ct_pt64(negative_self._raw, so._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            else:
                return ShellTensor64(
                    shell_ops.sub_pt_pt64(so._raw, self._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    False,
                )
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_encrypted:
            return ShellTensor64(
                shell_ops.neg_ct64(self._raw),
                self._context,
                self._num_slots,
                self._underlying_dtype,
                True,
            )
        else:
            return ShellTensor64(
                shell_ops.neg_pt64(self._context, self._raw),
                self._context,
                self._num_slots,
                self._underlying_dtype,
                False,
            )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            if self.is_encrypted and other.is_encrypted:
                return ShellTensor64(
                    shell_ops.mul_ct_ct64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.mul_ct_pt64(self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and other.is_encrypted:
                return ShellTensor64(
                    shell_ops.mul_ct_pt64(other._raw, self._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    True,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                return ShellTensor64(
                    shell_ops.mul_pt_pt64(self._context, self._raw, other._raw),
                    self._context,
                    self._num_slots,
                    self._underlying_dtype,
                    False,
                )
            else:
                raise ValueError("Invalid operands")
        elif isinstance(other, tf.Tensor):
            # TODO(jchoncholas): If scalar tensor, call special op, else
            # lift tensorflow tensor to shell tensor before add
            so = to_shell_tensor(self._context, other)
            return self * so

        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other


def _tensor_conversion_function(tensor, dtype=None, name=None, as_ref=False):
    if not name is None:
        return NotImplemented
    if as_ref:
        return NotImplemented
    if not dtype in [
        tf.uint8,
        tf.int8,
        tf.int16,
        tf.int32,
        tf.int64,
        tf.float32,
        tf.float64,
        None,
    ]:
        return NotImplemented
    return from_shell_tensor(tensor, dtype=dtype)


# Implicit conversion from ShellTensor64 to tensor requires decryption. Will
# raise ValueError if ShellTensor is still encrypted. Thus it may not actually
# be that useful.
tf.register_tensor_conversion_function(ShellTensor64, _tensor_conversion_function)


def to_shell_tensor(context, tensor):
    if isinstance(tensor, ShellTensor64):
        return tensor
    if isinstance(tensor, tf.Tensor):
        return ShellTensor64(
            shell_ops.polynomial_import64(context, tensor),
            context,
            tensor.shape[0],
            tensor.dtype,
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
    return shell_ops.polynomial_export64(
        s_tensor._context,
        s_tensor._raw,
        dtype=s_tensor._underlying_dtype,
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


def matmul(x, y, temp_key=None):
    if isinstance(x, ShellTensor64) and isinstance(y, tf.Tensor):
        return ShellTensor64(
            shell_ops.mat_mul_ct_pt64(x._raw, y),
            x._context,
            x._num_slots,
            x._underlying_dtype,
            is_enc=True,
        )
    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
        num_slots = y._num_slots
        context = y._context

        if y.is_encrypted:
            # TODO(jchoncholas): Requiring key is a limitation for now.
            # Once shell has galois key rotation we can compute a reduce operation
            # over a polynomial which is required for this operation. For now
            # we cheat and decrypt, compute, re-encrypt.
            assert temp_key is not None, "key must be provided for matmul_pt_ct"
            print(
                f"WARNING: matmul for pt*ct is not computed under encryption yet. Decrypting, computing, re-encrypting."
            )
            y = y.get_decrypted(temp_key)
        else:
            y = from_shell_tensor(y)

        y_orgig_dtype = y.dtype
        if not x.dtype is y.dtype:
            # TODO(jchoncholas): When shell support rotations, casting will not
            # be required
            print(
                f"WARNING: matmult requires casting y to x's dtype (from {y.dtype} to {x.dtype})"
            )
            y = tf.cast(y, x.dtype)

        res = tf.matmul(x, y)
        res = tf.expand_dims(res, axis=0)
        res = tf.repeat(res, repeats=[num_slots], axis=0)

        res = tf.cast(res, y_orgig_dtype)
        shell_res = to_shell_tensor(context, res)
        enc_res = shell_res.get_encrypted(temp_key)
        # dimensions of enc_res: num_slots by x_m by y_n
        # Dimensions match what would be returned when galois slot rotation
        # is implemented.

        return enc_res

    elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        return tf.matmul(x, y)

    else:
        raise ValueError(f"Unsupported types for matmul. Got {type(x)} and {type(y)}")
