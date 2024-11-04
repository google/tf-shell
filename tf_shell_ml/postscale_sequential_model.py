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
            if self.disable_encryption:
                enc_y = labels
            else:
                backprop_context = self.backprop_context_fn()
                secret_key = tf_shell.create_key64(backprop_context, self.cache_path)
                # Encrypt the batch of secret labels.
                enc_y = tf_shell.to_encrypted(labels, secret_key, backprop_context)

        with tf.device(self.jacobian_device):
            features = tf.identity(features)  # copy to GPU if needed

            # self.layers[-1].activation = tf.keras.activations.linear
            with tf.GradientTape(
                persistent=tf.executing_eagerly() or self.jacobian_pfor
            ) as tape:
                y_pred = self(features, training=True, with_softmax=False)

            grads = tape.jacobian(
                y_pred,
                self.trainable_variables,
                # unconnected_gradients=tf.UnconnectedGradients.ZERO, broken with pfor
                parallel_iterations=self.jacobian_pfor_iterations,
                experimental_use_pfor=self.jacobian_pfor,
            )
            # ^  layers list x (batch size x num output classes x weights) matrix
            # dy_pred_j/dW_sample_class

            # compute the
            # activation manually.
            y_pred = tf.nn.softmax(y_pred)

        with tf.device(self.features_party_dev):
            # Compute prediction - labels (where labels may be encrypted).
            scalars = enc_y.__rsub__(y_pred)  # dJ/dy_pred
            # ^  batch_size x num output classes.

            # Expand the last dim so that the subsequent multiplications are
            # broadcasted.
            scalars = tf_shell.expand_dims(scalars, axis=-1)
            # ^ batch_size x num output classes x 1

            # Flatten and remember the original shape of the gradient in order
            # to unpack them after the multiplication so they can be applied to
            # the model.
            grads, slot_size, grad_shapes, flattened_grad_shapes = (
                self.flatten_jacobian_list(grads)
            )
            # ^ batch_size x num output classes x all flattened weights

            max_two_norm = self.flat_jacobian_two_norm(grads)

            # Scale the gradients.
            grads = scalars * grads
            # ^ batch_size x num output classes x all flattened weights

            # Sum over the output classes.
            grads = tf_shell.reduce_sum(grads, axis=1)
            # ^ batch_size x all flattened weights

            # Recover the original shapes of the gradients.
            grads = self.unflatten_batch_grad_list(
                grads, slot_size, grad_shapes, flattened_grad_shapes
            )
            # ^ layers list (batch_size x weights)

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
                t = tf.cast(backprop_context.plaintext_modulus, tf.float32)
                t_half = t // 2
                mask_scaling_factors = [g._scaling_factor for g in grads]
                masks = [
                    tf.random.uniform(
                        tf_shell.shape(g),
                        dtype=tf.float32,
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
                    dtype=float,
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
                epsilon = tf.constant(1e-6, dtype=float)

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
                        loss = self.loss_fn(labels, y_pred)
                        metric.update_state(loss)
                    else:
                        # Loss is unknown when encrypted.
                        metric.update_state(0.0)
                else:
                    if self.disable_encryption:
                        metric.update_state(labels, y_pred)
                    else:
                        # Other metrics are uknown when encrypted.
                        zeros = tf.broadcast_to(0, tf.shape(y_pred))
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
