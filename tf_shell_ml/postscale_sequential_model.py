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

    def shell_train_step(self, features, labels, read_key_from_cache):
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_encryption:
                enc_y = labels
            else:
                backprop_context = self.backprop_context_fn(read_key_from_cache)
                secret_key = tf_shell.create_key64(backprop_context, self.cache_path)
                # Encrypt the batch of secret labels.
                enc_y = tf_shell.to_encrypted(labels, secret_key, backprop_context)

        predictions_list = []
        jacobians_list = []
        max_two_norms_list = []
        split_features, remainder = self.split_with_padding(
            features, len(self.jacobian_devices)
        )
        for i, d in enumerate(self.jacobian_devices):
            with tf.device(d):
                f = tf.identity(split_features[i])  # copy to GPU if needed
                prediction, jacobians = self.predict_and_jacobian(f)
                if i == len(self.jacobian_devices) - 1 and remainder > 0:
                    # The last device may have a remainder that was padded.
                    prediction = prediction[: features.shape[0] - remainder]
                    jacobians = jacobians[: features.shape[0] - remainder]
                predictions_list.append(prediction)
                jacobians_list.append(jacobians)
                max_two_norms_list.append(self.jacobian_max_two_norm(jacobians))

        with tf.device(self.features_party_dev):
            # For some reason, when running the jacobian on multiple devices,
            # the weights must be touched otherwise training loss goes to NaN.
            # Maybe it is to ensure the weights are on assigned to
            # features_party device when the final gradient is added to weights?
            _ = self(features, training=True, with_softmax=False)
            predictions = tf.concat(predictions_list, axis=0)
            max_two_norm = tf.reduce_max(max_two_norms_list)
            jacobains = [tf.concat(j, axis=0) for j in zip(*jacobians_list)]

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
                # Note, checking the gradients requires decryption on the
                # features party which breaks security of the protocol.
                bp_scaling_factors = [g._scaling_factor for g in grads]
                dec_grads = [tf_shell.to_tensorflow(g, secret_key) for g in grads]
                self.warn_on_overflow(
                    dec_grads,
                    bp_scaling_factors,
                    tf.identity(backprop_context.plaintext_modulus),
                    "WARNING: Backprop gradient may have overflowed.",
                )

            # Mask the encrypted gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads, masks, mask_scaling_factors = self.mask_gradients(
                    backprop_context, grads
                )

            if not self.disable_noise:
                # Features party encrypts the max two norm to send to the labels
                # party so they can scale the noise.
                noise_context = self.noise_context_fn(read_key_from_cache)
                noise_secret_key = tf_shell.create_key64(noise_context, self.cache_path)

                # The noise context must have the same number of slots
                # (encryption ring degree) as used in backpropagation.
                if not self.disable_encryption:
                    tf.assert_equal(
                        tf.identity(backprop_context.num_slots),
                        noise_context.num_slots,
                        message="Backprop and noise contexts must have the same number of slots.",
                    )

                # The noise scaling factor must always be 1. Encryptions already
                # have the scaling factor applied when the noise is applied,
                # and an additional noise scaling factor is not needed.
                tf.assert_equal(
                    noise_context.scaling_factor,
                    1,
                    message="Noise scaling factor must be 1.",
                )

                # Compute the noise factors for the distributed noise sampling
                # sub-protocol.
                enc_a, enc_b = self.compute_noise_factors(
                    noise_context, noise_secret_key, max_two_norm
                )

        with tf.device(self.labels_party_dev):
            if not self.disable_encryption:
                # Decrypt the weight gradients.
                grads = [tf_shell.to_tensorflow(g, secret_key) for g in grads]

            # Sum the masked gradients over the batch.
            if self.disable_masking or self.disable_encryption:
                # No mask has been added so a only a normal sum is required.
                grads = [tf.reduce_sum(g, 0) for g in grads]
            else:
                grads = [
                    tf_shell.reduce_sum_with_mod(g, 0, backprop_context, s)
                    for g, s in zip(grads, mask_scaling_factors)
                ]

            if not self.disable_noise:
                # Efficiently pack the masked gradients to prepare for
                # encryption. This is special because the masked gradients are
                # no longer batched so the packing must be done manually.
                (flat_grads, grad_shapes, flattened_grad_shapes, total_grad_size) = (
                    self.flatten_and_pad_grad_list(
                        grads, tf.identity(noise_context.num_slots)
                    )
                )

                # Add the encrypted noise to the masked gradients.
                grads = self.noise_gradients(noise_context, flat_grads, enc_a, enc_b)

        with tf.device(self.features_party_dev):
            if not self.disable_noise:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = tf_shell.to_tensorflow(grads, noise_secret_key)

                # Unpack the noise after decryption.
                grads = self.unflatten_and_unpad_grad(
                    flat_grads,
                    grad_shapes,
                    flattened_grad_shapes,
                    total_grad_size,
                )

            # Unmask the gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads = self.unmask_gradients(
                    backprop_context, grads, masks, mask_scaling_factors
                )

            if not self.disable_noise:
                if self.check_overflow_INSECURE:
                    nosie_scaling_factor = grads._scaling_factor
                    self.warn_on_overflow(
                        [flat_grads],
                        [nosie_scaling_factor],
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )

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
                None
                if self.disable_encryption
                else tf.identity(backprop_context.num_slots)
            )
