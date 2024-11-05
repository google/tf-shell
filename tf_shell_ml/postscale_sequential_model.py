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
import tensorflow.keras as keras
import tf_shell
from tf_shell_ml.model_base import SequentialBase


class PostScaleSequential(SequentialBase):
    def __init__(
        self,
        layers,
        *args,
        **kwargs,
    ):
        super().__init__(layers, *args, **kwargs)

        if len(self.layers) > 0:
            self.layers[0].is_first_layer = True

            # Override the activation of the last layer. Set it to linear so
            # when the jacobian is computed, the derivative of the activation
            # function is a no-op. This is required for the post scale protocol.
            self.layers[-1].activation = tf.keras.activations.linear

    def call(self, inputs, training=False, with_softmax=True):
        prediction = super().call(inputs, training)
        if not with_softmax:
            return prediction
        # Perform the last layer activation since it is removed for training
        # purposes.
        return tf.nn.softmax(prediction)

    def shell_train_step(self, features, labels):
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_encryption:
                enc_y = labels
            else:
                backprop_context = self.backprop_context_fn()
                secret_key = tf_shell.create_key64(backprop_context, self.cache_path)
                # Encrypt the batch of secret labels.
                enc_y = tf_shell.to_encrypted(labels, secret_key, backprop_context)

        with tf.device(self.jacobian_device):
            features = tf.identity(features)  # copy to GPU if needed
            predictions, jacobians = self.predict_and_jacobian(features)
            max_two_norm = self.jacobian_max_two_norm(jacobians)

        with tf.device(self.features_party_dev):
            # Compute prediction - labels (where labels may be encrypted).
            scalars = enc_y.__rsub__(predictions)  # dJ/dprediction
            # ^  batch_size x num output classes.

            # Scale each gradient. Since 'scalars' may be a vector of
            # ciphertexts, this requires multiplying plaintext gradient for the
            # specific layer (2d) by the ciphertext (scalar).
            grads = []
            for j in jacobians:
                # Ignore the batch size and num output classes dimensions and
                # recover just the number of dimensions in the weights.
                num_weight_dims = len(j.shape) - 2

                # Make the scalars the same shape as the gradients so the
                # multiplication can be broadcasted. Doing this inside the loop
                # is okay, TensorFlow will reuse the same expanded tensor if
                # their dimensions match across iterations.
                scalars_exp = scalars
                for _ in range(num_weight_dims):
                    scalars_exp = tf_shell.expand_dims(scalars_exp, axis=-1)

                # Scale the jacobian.
                scaled_grad = scalars_exp * j
                # ^ batch_size x num output classes x weights

                # Sum over the output classes. At this point, this is a gradient
                # and no longer a jacobian.
                scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=1)
                # ^  batch_size x weights

                grads.append(scaled_grad)

            # Check if the post-scaled gradients overflowed.
            if not self.disable_encryption and self.check_overflow_INSECURE:
                # Note, checking the backprop gradients requires decryption
                # on the features party which breaks security of the protocol.
                bp_scaling_factors = [g._scaling_factor for g in grads]
                dec_grads = [tf_shell.to_tensorflow(g, secret_key) for g in grads]
                self.warn_on_overflow(
                    dec_grads,
                    bp_scaling_factors,
                    backprop_context.plaintext_modulus,
                    "WARNING: Backprop gradient may have overflowed.",
                )

            # Mask the encrypted grads to prepare for decryption. The masks may
            # overflow during the reduce_sum over the batch. When the masks are
            # operated on, they are multiplied by the scaling factor, so it is
            # not necessary to mask the full range -t/2 to t/2. (Though it is
            # possible, it unnecessarily introduces noise into the ciphertext.)
            if not self.disable_masking and not self.disable_encryption:
                t = tf.cast(
                    backprop_context.plaintext_modulus, tf.keras.backend.floatx()
                )
                t_half = t // 2
                mask_scaling_factors = [g._scaling_factor for g in grads]
                masks = [
                    tf.random.uniform(
                        tf_shell.shape(g),
                        dtype=tf.keras.backend.floatx(),
                        minval=-t_half / s,
                        maxval=t_half / s,
                    )
                    for g, s in zip(grads, mask_scaling_factors)
                ]

                # Mask the encrypted gradients to prepare for decryption.
                grads = [g + m for g, m in zip(grads, masks)]

            if not self.disable_noise:
                # Features party encrypts the max two norm to send to the labels
                # party so they can scale the noise.
                noise_context = self.noise_context_fn()
                noise_secret_key = tf_shell.create_key64(noise_context, self.cache_path)
                max_two_norm = tf.expand_dims(max_two_norm, 0)
                max_two_norm = tf.repeat(max_two_norm, noise_context.num_slots, axis=0)
                enc_max_two_norm = tf_shell.to_encrypted(
                    max_two_norm, noise_secret_key, noise_context
                )

        with tf.device(self.labels_party_dev):
            if not self.disable_encryption:
                # Decrypt the weight gradients.
                grads = [tf_shell.to_tensorflow(g, secret_key) for g in grads]

            # Sum the masked gradients over the batch.
            if not self.disable_masking and not self.disable_encryption:
                grads = [
                    tf_shell.reduce_sum_with_mod(g, 0, backprop_context, s)
                    for g, s in zip(grads, mask_scaling_factors)
                ]
            else:
                grads = [tf.reduce_sum(g, 0) for g in grads]

            if not self.disable_noise:
                if not self.disable_encryption:
                    tf.assert_equal(
                        backprop_context.num_slots,
                        noise_context.num_slots,
                        message="Backprop and noise contexts must have the same number of slots.",
                    )

                # Efficiently pack the masked gradients to prepare for
                # encryption. This is special because the masked gradients are
                # no longer batched so the packing must be done manually.
                (flat_grads, grad_shapes, flattened_grad_shapes, total_grad_size) = (
                    self.flatten_and_pad_grad_list(grads, noise_context.num_slots)
                )

                # Sample the noise
                noise = tf.random.normal(
                    tf.shape(flat_grads),
                    stddev=self.noise_multiplier,
                    dtype=tf.keras.backend.floatx(),
                )
                # Scale it by the encrypted max two norm.
                enc_noise = enc_max_two_norm * noise
                # Add the encrypted noise to the flat gradients.
                grads = enc_noise + flat_grads

        with tf.device(self.features_party_dev):
            if not self.disable_noise:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = tf_shell.to_tensorflow(grads, noise_secret_key)

                if self.check_overflow_INSECURE:
                    nosie_scaling_factor = grads._scaling_factor
                    self.warn_on_overflow(
                        [flat_grads],
                        [nosie_scaling_factor],
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )
                # Unpack the noise after decryption.
                grads = self.unflatten_and_unpad_grad(
                    flat_grads,
                    grad_shapes,
                    flattened_grad_shapes,
                    total_grad_size,
                )

            if not self.disable_masking and not self.disable_encryption:
                # Sum the masks over the batch.
                sum_masks = [
                    tf_shell.reduce_sum_with_mod(m, 0, backprop_context, s)
                    for m, s in zip(masks, mask_scaling_factors)
                ]

                # Unmask the batch gradient.
                grads = [mg - m for mg, m in zip(grads, sum_masks)]

                # SHELL represents floats as integers between [0, t) where t is
                # the plaintext modulus. To mimic SHELL's modulo operations in
                # TensorFlow, numbers which exceed the range [-t/2, t/2] are
                # shifted back into the range.
                epsilon = tf.constant(1e-6, dtype=tf.keras.backend.floatx())

                def rebalance(x, s):
                    r_bound = t_half / s + epsilon
                    l_bound = -t_half / s - epsilon
                    t_over_s = t / s
                    x = tf.where(x > r_bound, x - t_over_s, x)
                    x = tf.where(x < l_bound, x + t_over_s, x)
                    return x

                grads = [rebalance(g, s) for g, s in zip(grads, mask_scaling_factors)]

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(grads, self.weights))

            for metric in self.metrics:
                if metric.name == "loss":
                    if self.disable_encryption:
                        loss = self.compiled_loss(labels, predictions)
                        metric.update_state(loss)
                    else:
                        # Loss is unknown when encrypted.
                        metric.update_state(0.0)
                else:
                    if self.disable_encryption:
                        metric.update_state(labels, predictions)
                    else:
                        # Other metrics are uknown when encrypted.
                        zeros = tf.broadcast_to(0, tf.shape(predictions))
                        metric.update_state(zeros, zeros)

            metric_results = {m.name: m.result() for m in self.metrics}

            # TensorFlow 2.18.0 added a "CompiledMetrics" metric which holds metrics
            # passed to compile in it's own dictionary. Keras wants all metrics to
            # be returned as a flat dictionary. Here we flatten the dictionary.
            result = {}
            for key, value in metric_results.items():
                if isinstance(value, dict):
                    result.update(value)  # add subdict directly into the dict
                else:
                    result[key] = value  # non-subdict elements are just copied

            return result, (
                None if self.disable_encryption else backprop_context.num_slots
            )
