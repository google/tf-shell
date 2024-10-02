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


def create_key64(context, cache_path=""):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64")

    @tf.py_function(Tout=tf.variant)
    def read_or_generate_keys(cache_path, context):
        exists = tf.io.gfile.exists(cache_path.numpy())
        if exists:
            cached_keys = tf.io.read_file(cache_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return cached_keys

        else:
            # num_keys = context.level
            # keys = tf.TensorArray(tf.variant, size=context.level, clear_after_read=False)

            # # Generate and store the first key in the last index.
            # keys = keys.write(
            #     num_keys - 1, shell_ops.key_gen64(context._get_context_at_level(num_keys))
            # )

            # # Mod reduce to compute the remaining keys.
            # keys, _ = tf.while_loop(
            #     lambda ks, l: l > 1,
            #     lambda ks, l: (
            #         ks.write(
            #             l - 2,
            #             shell_ops.modulus_reduce_key64(
            #                 context._get_context_at_level(l), ks.read(l - 1)
            #             ),
            #         ),
            #         l - 1,
            #     ),
            #     loop_vars=[keys, num_keys],
            #     shape_invariants=[
            #         tf.TensorSpec(None, dtype=tf.variant),
            #         tf.TensorSpec([], dtype=tf.int32),
            #     ],
            #     parallel_iterations=1,
            # )

            # gathered_keys = keys.gather(tf.range(0, num_keys))

            # if cache_path is not None:
            #     print(f"Writing keys to {cache_path}")
            #     tf.io.write_file(cache_path, tf.io.serialize_tensor(gathered_keys))

            # return gathered_keys

            # The approach above is graph-friendly but this is unnecessary when
            # wrapped with a tf.py_function. The code below is more concise and
            # slightly faster.
            num_keys = context.level.numpy()
            keys = [shell_ops.key_gen64(context._get_context_at_level(num_keys))]
            for i in range(num_keys, 1, -1):
                keys.insert(
                    0,
                    shell_ops.modulus_reduce_key64(
                        context._raw_contexts[i - 1], keys[0]
                    ),
                )

            if cache_path.numpy() != b"":
                tf.io.write_file(cache_path, tf.io.serialize_tensor(keys))

            return tf.convert_to_tensor(keys, dtype=tf.variant)

    raw_keys = read_or_generate_keys(cache_path, context)
    return ShellKey64(_raw_keys_at_level=raw_keys)


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


def create_rotation_key64(context, key, cache_path=""):
    """Create rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys to perform an arbitrary number of slot rotations.
    Since rotation key generation is expensive, the caller can choose to skip
    generating keys at levels (particular number of moduli) at which no
    rotations are required."""
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    @tf.py_function(Tout=tf.variant)
    def read_or_generate_keys(context, key, cache_path):
        exists = tf.io.gfile.exists(cache_path.numpy())
        if exists:
            cached_keys = tf.io.read_file(cache_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return cached_keys

        else:
            rot_keys = []
            for i in range(context.level.numpy(), 0, -1):
                rot_keys.insert(
                    0,
                    shell_ops.rotation_key_gen64(
                        context._raw_contexts[i - 1], key._raw_keys_at_level[i - 1]
                    ),
                )

            if cache_path.numpy() != b"":
                tf.io.write_file(cache_path, tf.io.serialize_tensor(rot_keys))

            return tf.convert_to_tensor(rot_keys, dtype=tf.variant)

    raw_keys = read_or_generate_keys(context, key, cache_path)
    return ShellRotationKey64(_raw_keys_at_level=raw_keys)


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


def create_fast_rotation_key64(context, key, cache_path=""):
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

    @tf.py_function(Tout=tf.variant)
    def read_or_generate_keys(context, key, cache_path):
        exists = tf.io.gfile.exists(cache_path.numpy())
        if exists:
            cached_keys = tf.io.read_file(cache_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return cached_keys

        else:
            rot_keys = []
            for i in range(context.level, 0, -1):
                rot_keys.insert(
                    0,
                    shell_ops.fast_rotation_key_gen64(
                        context._raw_contexts[i - 1], key._raw_keys_at_level[i - 1]
                    ),
                )

            if cache_path.numpy() != b"":
                tf.io.write_file(cache_path, tf.io.serialize_tensor(rot_keys))

            return tf.convert_to_tensor(rot_keys, dtype=tf.variant)

    raw_keys = read_or_generate_keys(context, key, cache_path)
    return ShellFastRotationKey64(_raw_keys_at_level=raw_keys)
