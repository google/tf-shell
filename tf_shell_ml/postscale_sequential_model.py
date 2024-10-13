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
        backprop_context_fn,
        noise_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        noise_multiplier=1.0,
        disable_encryption=False,
        disable_masking=False,
        disable_noise=False,
        check_overflow=False,
        cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(layers, *args, **kwargs)

        if len(self.layers) > 0:
            self.layers[0].is_first_layer = True
            # Do not set the derivative of the activation function for the last
            # layer in the model. The derivative of the categorical crossentropy
            # loss function times the derivative of a softmax is just y_pred - y
            # (which is much easier to compute than each of them individually).
            # So instead just let the loss function derivative incorporate
            # y_pred - y and let the derivative of this last layer's activation
            # be a no-op.
            self.layers[-1].activation_deriv = None

        self.backprop_context_fn = backprop_context_fn
        self.noise_context_fn = noise_context_fn
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.clipping_threshold = 10000000
        self.context_prepped = False
        self.disable_encryption = disable_encryption
        self.noise_multiplier = noise_multiplier
        self.disable_masking = disable_masking
        self.disable_noise = disable_noise
        self.check_overflow = check_overflow
        self.cache_path = cache_path

    def compile(self, optimizer, shell_loss, loss, metrics=[], **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        if shell_loss is None:
            raise ValueError("shell_loss must be provided")
        self.loss_fn = shell_loss

        # Keras ignores metrics that are not used during training. When training
        # with encryption, the metrics are not updated. Store the metrics so
        # they can be recovered during validation.
        self.val_metrics = metrics

    def _unpack(self, x_list):
        batch_size = tf.shape(x_list[0])[0] // 2
        return [x[0] + x[batch_size] for x in x_list]

    def train_step(self, data):
        x, y = data

        with tf.device(self.labels_party_dev):
            if self.disable_encryption:
                enc_y = y
            else:
                backprop_context = self.backprop_context_fn()
                secret_key = tf_shell.create_key64(backprop_context, self.cache_path)
                # Encrypt the batch of secret labels y.
                enc_y = tf_shell.to_encrypted(y, secret_key, backprop_context)

        with tf.device(self.features_party_dev):
            # Unset the activation function for the last layer so it is not used in
            # computing the gradient. The effect of the last layer activation function
            # is factored out of the gradient computation and accounted for below.
            self.layers[-1].activation = tf.keras.activations.linear

            with tf.GradientTape(persistent=tf.executing_eagerly()) as tape:
                y_pred = self(x, training=True)  # forward pass
            grads = tape.jacobian(
                y_pred,
                self.trainable_variables,
                parallel_iterations=1,
                experimental_use_pfor=False,
            )
            # grads = tape.jacobian(y_pred, self.trainable_variables, experimental_use_pfor=True)
            # ^  layers list x (batch size x num output classes x weights) matrix
            # dy_pred_j/dW_sample_class

            # Reset the activation function for the last layer and compute the real
            # prediction.
            self.layers[-1].activation = tf.keras.activations.sigmoid
            y_pred = self(x, training=False)

            # Compute y_pred - y (where y may be encrypted).
            # scalars = y_pred - y  # dJ/dy_pred
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
            if self.disable_masking or self.disable_encryption:
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
                grads = tf_shell.to_tensorflow(grads, noise_secret_key)
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

                # SHELL represents floats as integers between [0, t) where t is the
                # plaintext modulus. To mimic SHELL's modulo operations in
                # TensorFlow, numbers which exceed the range [-t/2, t/2] are shifted
                # back into the range.
                epsilon = tf.constant(1e-6, dtype=float)

                def rebalance(x, s):
                    r_bound = t_half / s + epsilon
                    l_bound = -t_half / s - epsilon
                    t_over_s = t / s
                    x = tf.where(x > r_bound, x - t_over_s, x)
                    x = tf.where(x < l_bound, x + t_over_s, x)
                    return x

                grads = [rebalance(g, s) for g, s in zip(grads, mask_scaling_factors)]

            if not self.disable_encryption and self.check_overflow:
                # If the unmasked gradient is between [-t/2, -t/4] or
                # [t/4, t/2], the gradient may have overflowed. Note this must
                # also take the scaling factor into account.
                overflowed = [
                    tf.abs(g) > t_half / 2 / s
                    for g, s in zip(grads, mask_scaling_factors)
                ]
                overflowed = [tf.reduce_any(o) for o in overflowed]
                overflowed = tf.reduce_any(overflowed)
                tf.cond(
                    overflowed,
                    lambda: tf.print("Gradient may have overflowed"),
                    lambda: tf.identity(overflowed),
                )

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(grads, self.weights))

        # Do not update metrics during secure training.
        if self.disable_encryption:
            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    loss = self.loss_fn(y, y_pred)
                    metric.update_state(loss)
                else:
                    metric.update_state(y, y_pred)

            metric_results = {m.name: m.result() for m in self.metrics}
        else:
            metric_results = {"num_slots": backprop_context.num_slots}

        return metric_results
