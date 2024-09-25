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
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        level -= 1  # Keys tensor start at level 1.
        tf.Assert(level >= 0, [f"level must be >= 0. Got {level}"])
        tf.Assert(
            level < tf.shape(self._raw_keys_at_level)[0],
            [f"level must be < {tf.shape(self._raw_keys_at_level)[0]}. Got {level}"],
        )
        return self._raw_keys_at_level[level]


def create_key64(context):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64")
    num_keys = context.level
    keys = tf.TensorArray(tf.variant, size=context.level, clear_after_read=False)

    # Generate and store the first key in the last index.
    keys = keys.write(context.level - 1, shell_ops.key_gen64(context._raw_context))

    # Mod reduce to compute the remaining keys.
    keys, context = tf.while_loop(
        lambda ks, c: c.level > 2,
        lambda ks, c: (
            ks.write(
                c.level - 2,
                shell_ops.modulus_reduce_key64(c._raw_context, ks.read(c.level - 1)),
            ),
            mod_reduce_context64(c),
        ),
        loop_vars=[keys, context],
        shape_invariants=[
            tf.TensorSpec(None, dtype=tf.variant),
            context._get_generic_context_spec(),
        ],
    )

    # Store the first key for level 1.
    keys = tf.cond(
        context.level == 2,
        lambda: keys.write(
            context.level - 2,
            shell_ops.modulus_reduce_key64(
                context._raw_context, keys.read(context.level - 1)
            ),
        ),
        lambda: keys,
    )

    return ShellKey64(_raw_keys_at_level=keys.gather(tf.range(0, num_keys)))


class ShellRotationKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        level -= 1  # Keys tensor start at level 1.
        tf.Assert(level >= 0, [f"level must be >= 0. Got {level}"])
        tf.Assert(
            level < tf.shape(self._raw_keys_at_level)[0],
            [f"level must be < {tf.shape(self._raw_keys_at_level)[0]}. Got {level}"],
        )
        return self._raw_keys_at_level[level]


def create_rotation_key64(context, key):
    """Create rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys to perform an arbitrary number of slot rotations.
    Since rotation key generation is expensive, the caller can choose to skip
    generating keys at levels (particular number of moduli) at which no
    rotations are required."""
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    num_keys = context.level
    rot_keys = tf.TensorArray(
        tf.variant,
        size=context.level,
        clear_after_read=False,
        infer_shape=False,
        element_shape=(),
    )

    # Generate rotation keys starting from the highest level.
    rot_keys, context = tf.while_loop(
        lambda ks, c: c.level > 1,
        lambda ks, c: (
            ks.write(
                c.level - 1,
                shell_ops.rotation_key_gen64(
                    c._raw_context, key._get_key_at_level(c.level)
                ),
            ),
            mod_reduce_context64(c),
        ),
        loop_vars=[rot_keys, context],
        shape_invariants=[
            tf.TensorSpec(None, dtype=tf.variant),
            context._get_generic_context_spec(),
        ],
    )

    # Store the first key for level 1.
    rot_keys = tf.cond(
        context.level == 1,
        lambda: rot_keys.write(
            context.level - 1,
            shell_ops.rotation_key_gen64(
                context._raw_context, key._get_key_at_level(context.level)
            ),
        ),
        lambda: rot_keys,
    )

    return ShellRotationKey64(_raw_keys_at_level=rot_keys.gather(tf.range(0, num_keys)))


class ShellFastRotationKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        level -= 1  # Keys tensor start at level 1.
        tf.Assert(level >= 0, [f"level must be >= 0. Got {level}"])
        tf.Assert(
            level < tf.shape(self._raw_keys_at_level)[0],
            [f"level must be < {tf.shape(self._raw_keys_at_level)[0]}. Got {level}"],
        )
        return self._raw_keys_at_level[level]


def create_fast_rotation_key64(context, key):
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

    num_keys = context.level
    rot_keys = tf.TensorArray(
        tf.variant,
        size=context.level,
        clear_after_read=False,
        infer_shape=False,
        element_shape=(),
    )

    # Generate rotation keys starting from the highest level.
    rot_keys, context = tf.while_loop(
        lambda ks, c: c.level > 1,
        lambda ks, c: (
            ks.write(
                c.level - 1,
                shell_ops.fast_rotation_key_gen64(
                    c._raw_context, key._get_key_at_level(c.level)
                ),
            ),
            mod_reduce_context64(c),
        ),
        loop_vars=[rot_keys, context],
        shape_invariants=[
            tf.TensorSpec(None, dtype=tf.variant),
            context._get_generic_context_spec(),
        ],
    )

    # Store the first key for level 1.
    rot_keys = tf.cond(
        context.level == 1,
        lambda: rot_keys.write(
            context.level - 1,
            shell_ops.fast_rotation_key_gen64(
                context._raw_context, key._get_key_at_level(context.level)
            ),
        ),
        lambda: rot_keys,
    )

    return ShellFastRotationKey64(
        _raw_keys_at_level=rot_keys.gather(tf.range(0, num_keys))
    )
