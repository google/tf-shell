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
from tf_shell.python.shell_context import ShellContext64
from tf_shell.python.shell_context import mod_reduce_context64
import tensorflow as tf
import typing


class ShellKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: typing.Mapping[int, tf.Tensor]

    def _get_key_at_level(self, level):
        if level not in self._raw_keys_at_level:
            raise ValueError(f"No key at level {level}.")
        return self._raw_keys_at_level[level]


def mod_reduce_key64(unreduced_context, raw_key):
    if not isinstance(unreduced_context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(raw_key, tf.Tensor):
        raise ValueError("raw_key must be a Tensor")


def create_key64(context, skip_at_mul_depth=[]):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64")

    raw_keys_at_level = {}

    # Generate and store the first key.
    key = shell_ops.key_gen64(context._raw_context)
    raw_keys_at_level[context.level] = key

    # Mod reduce to compute the remaining keys.
    while context.level > 1:
        key = shell_ops.modulus_reduce_key64(context._raw_context, key)
        context = mod_reduce_context64(context)

        if context.level not in skip_at_mul_depth:
            raw_keys_at_level[context.level] = key

    return ShellKey64(_raw_keys_at_level=raw_keys_at_level)


class ShellRotationKey64(tf.experimental.ExtensionType):
    _raw_rot_keys_at_level: typing.Mapping[int, tf.Tensor]

    def _get_key_at_level(self, level):
        if level not in self._raw_rot_keys_at_level:
            raise ValueError(f"No rotation key at level {level}.")
        return self._raw_rot_keys_at_level[level]


def create_rotation_key64(context, key, skip_at_mul_depth=[]):
    """Create rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys to perform an arbitrary number of slot rotations.
    Since rotation key generation is expensive, the caller can choose to skip
    generating keys at levels (particular number of moduli) at which no
    rotations are required."""
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    raw_rot_keys_at_level = {}
    while context.level >= 0:
        if context.level not in skip_at_mul_depth:
            raw_rot_keys_at_level[context.level] = shell_ops.rotation_key_gen64(
                context._raw_context,
                key._get_key_at_level(context.level),
            )

        if context.level <= 1:
            break

        context = mod_reduce_context64(context)

    return ShellRotationKey64(_raw_rot_keys_at_level=raw_rot_keys_at_level)


class ShellFastRotationKey64(tf.experimental.ExtensionType):
    _raw_rot_keys_at_level: typing.Mapping[int, tf.Tensor]

    def _get_key_at_level(self, level):
        if level not in self._raw_rot_keys_at_level:
            raise ValueError(f"No rotation key at level {level}.")
        return self._raw_rot_keys_at_level[level]


def create_fast_rotation_key64(context, key, skip_at_mul_depth=[]):
    """Create fast rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys *decrypt* a previously "fast" rotated ciphertext.
    These keys are much faster to generated than regular rotation keys, and
    they are not needed for the rotation operation itself, only for decryption.
    Fast rotation is only supported for degree 1 ciphertexts.
    """
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    raw_rot_keys_at_level = {}
    while context.level >= 0:
        if context.level not in skip_at_mul_depth:
            raw_rot_keys_at_level[context.level] = shell_ops.fast_rotation_key_gen64(
                context._raw_context, key._get_key_at_level(context.level)
            )

        if context.level <= 1:
            break

        context = mod_reduce_context64(context)

    return ShellFastRotationKey64(_raw_rot_keys_at_level=raw_rot_keys_at_level)
