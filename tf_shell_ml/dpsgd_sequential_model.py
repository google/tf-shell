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
import tf_shell_ml
from tf_shell_ml.model_base import SequentialBase


class DpSgdSequential(SequentialBase):
    def __init__(
        self,
        layers,
        *args,
        **kwargs,
    ):
        super().__init__(layers, *args, **kwargs)

        if len(self.layers) > 0:
            self.layers[0].is_first_layer = True
            # Do not set the the activation function for the last layer in the
            # model. The derivative of the categorical crossentropy loss
            # function times the derivative of a softmax is just predictions - labels
            # (which is much easier to compute than each of them individually).
            # So instead just let the loss function derivative incorporate
            # predictions - labels and let the derivative of this last layer's
            # activation be a no-op.
            self.layers[-1].activation = None
            self.layers[-1].activation_deriv = None

    def call(self, features, training=False, with_softmax=True):
        predictions = features
        for l in self.layers:
            predictions = l(predictions, training=training)

        if not with_softmax:
            return predictions
        # Perform the last layer activation since it is removed for training
        # purposes.
        return tf.nn.softmax(predictions)

    def shell_train_step(self, features, labels):
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_encryption:
                enc_y = labels
            else:
                backprop_context = self.backprop_context_fn()
                backprop_secret_key = tf_shell.create_key64(
                    backprop_context, self.cache_path
                )
                # Encrypt the batch of secret labels.
                enc_y = tf_shell.to_encrypted(
                    labels, backprop_secret_key, backprop_context
                )

        predictions_list = []
        max_two_norms_list = []
        split_features, remainder = self.split_with_padding(
            features, len(self.jacobian_devices)
        )
        for i, d in enumerate(self.jacobian_devices):
            with tf.device(d):
                f = tf.identity(split_features[i])  # copy to GPU if needed
                prediction, jacobians = self.predict_and_jacobian(
                    f,
                    skip_jacobian=self.disable_noise,  # Jacobian only needed for noise.
                )
                if i == len(self.jacobian_devices) - 1 and remainder > 0:
                    # The last device may have a remainder that was padded.
                    prediction = prediction[: features.shape[0] - remainder]
                    jacobians = jacobians[: features.shape[0] - remainder]
                predictions_list.append(prediction)
                max_two_norms_list.append(self.jacobian_max_two_norm(jacobians))

        with tf.device(self.features_party_dev):
            # For some reason, when running the jacobian on multiple devices,
            # the weights must be touched otherwise training loss goes to NaN.
            # Maybe it is to ensure the weights are on assigned to
            # features_party device when the final gradient is added to weights?
            _ = self(features, training=True, with_softmax=False)
            predictions = tf.concat(predictions_list, axis=0)
            max_two_norm = tf.reduce_max(max_two_norms_list)

            # Backward pass.
            dx = enc_y.__rsub__(predictions)  # Derivative of CCE loss and softmax.
            dJ_dw = []  # Derivatives of the loss with respect to the weights.
            dJ_dx = [dx]  # Derivatives of the loss with respect to the inputs.
            for l in reversed(self.layers):
                dw, dx = l.backward(dJ_dx[-1])
                dJ_dw.extend(dw)
                dJ_dx.append(dx)

            # Check if the backproped gradients overflowed.
            if not self.disable_encryption and self.check_overflow_INSECURE:
                # Note, checking the backprop gradients requires decryption
                # on the features party which breaks security of the protocol.
                bp_scaling_factors = [g._scaling_factor for g in dJ_dw]
                dec_dJ_dw = [
                    tf_shell.to_tensorflow(g, backprop_secret_key) for g in dJ_dw
                ]
                self.warn_on_overflow(
                    dec_dJ_dw,
                    bp_scaling_factors,
                    backprop_context.plaintext_modulus,
                    "WARNING: Backprop gradient may have overflowed.",
                )

            # Mask the encrypted grads to prepare for decryption. The masks may
            # overflow during the reduce_sum over the batch. When the masks are
            # operated on, they are multiplied by the scaling factor, so it is
            # not necessary to mask the full range -t/2 to t/2. (Though it is
            # possible, it unnecessarily introduces noise into the ciphertext.)
            if self.disable_masking or self.disable_encryption:
                grads = [g for g in reversed(dJ_dw)]
            else:
                t = tf.cast(
                    backprop_context.plaintext_modulus, tf.keras.backend.floatx()
                )
                t_half = t // 2
                mask_scaling_factors = [g._scaling_factor for g in reversed(dJ_dw)]
                masks = [
                    tf.random.uniform(
                        tf_shell.shape(g),
                        dtype=tf.keras.backend.floatx(),
                        minval=-t_half / s,
                        maxval=t_half / s,
                    )
                    for g, s in zip(reversed(dJ_dw), mask_scaling_factors)
                ]

                # Mask the encrypted gradients and reverse the order to match
                # the order of the layers.
                grads = [(g + m) for g, m in zip(reversed(dJ_dw), masks)]

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
                # Decrypt the weight gradients with the backprop key.
                grads = [tf_shell.to_tensorflow(g, backprop_secret_key) for g in grads]

            # Sum the masked gradients over the batch.
            if self.disable_encryption or self.disable_masking:
                # No mask, only sum required.
                grads = [tf.reduce_sum(g, axis=0) for g in grads]
            else:
                grads = [
                    tf_shell.reduce_sum_with_mod(g, 0, backprop_context, s)
                    for g, s in zip(grads, mask_scaling_factors)
                ]

            if not self.disable_noise:
                if not self.disable_encryption:
                    tf.assert_equal(
                        backprop_context.num_slots,
                        noise_context.num_slots,
                        message="Backprop and noise contexts must have the same number of slots.",
                    )

                # Efficiently pack the masked gradients to prepare for adding
                # the encrypted noise. This is special because the masked
                # gradients are no longer batched, so the packing must be done
                # manually.
                (
                    flat_grads,
                    grad_shapes,
                    flattened_grad_shapes,
                    total_grad_size,
                ) = self.flatten_and_pad_grad_list(grads, noise_context.num_slots)

                # Sample the noise.
                noise = tf.random.normal(
                    tf.shape(flat_grads),
                    stddev=self.noise_multiplier,
                    dtype=tf.keras.backend.floatx(),
                )
                # Scale it by the encrypted max two norm.
                enc_noise = enc_max_two_norm * noise
                # Add the encrypted noise to the flat masked gradients.
                grads = enc_noise + flat_grads

        with tf.device(self.features_party_dev):
            if not self.disable_noise:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = tf_shell.to_tensorflow(grads, noise_secret_key)

                if self.check_overflow_INSECURE:
                    nosie_scaling_factors = grads._scaling_factor
                    self.warn_on_overflow(
                        [flat_grads],
                        [nosie_scaling_factors],
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )

                # Unpack the noised grads after decryption.
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
                # shifted back into the range. Note, this could be done as a
                # custom op with tf-shell, but this is simpler for now.
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
