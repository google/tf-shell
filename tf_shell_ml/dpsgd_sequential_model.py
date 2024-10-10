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
import tempfile
from tf_shell_ml.model_base import SequentialBase


class DpSgdSequential(SequentialBase):
    def __init__(
        self,
        layers,
        shell_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        needs_public_rotation_key=False,
        noise_multiplier=1.0,
        disable_encryption=False,
        disable_masking=False,
        disable_noise=False,
        check_overflow_close=True,
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

        self.shell_context_fn = shell_context_fn
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.needs_public_rotation_key = needs_public_rotation_key
        self.noise_multiplier = noise_multiplier
        self.disable_encryption = disable_encryption
        self.disable_masking = disable_masking
        self.disable_noise = disable_noise
        self.check_overflow_close = check_overflow_close

    def compile(self, optimizer, shell_loss, loss, metrics=[], **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        if shell_loss is None:
            raise ValueError("shell_loss must be provided")
        self.loss_fn = shell_loss

        # Keras ignores metrics that are not used during training. When training
        # with encryption, the metrics are not updated. Store the metrics so
        # they can be recovered during validation.
        self.val_metrics = metrics

    def call(self, x, training=False):
        for l in self.layers:
            x = l(x, training=training)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.unpacking_funcs = []
        for l in self.layers:
            if hasattr(l, "unpacking_funcs"):
                self.unpacking_funcs.extend(l.unpacking_funcs())

    def compute_max_two_norm_and_pred(self, features):
        with tf.GradientTape(persistent=tf.executing_eagerly()) as tape:
            y_pred = self(features, training=True)  # forward pass
        grads = tape.jacobian(
            y_pred,
            self.trainable_variables,
            parallel_iterations=1,
            experimental_use_pfor=False,
        )
        # ^  layers list x (batch size x num output classes x weights) matrix

        if len(grads) == 0:
            raise ValueError("No gradients found")
        slot_size = tf.shape(grads[0])[0]
        num_output_classes = grads[0].shape[1]

        flat_grads = [tf.reshape(g, [slot_size, num_output_classes, -1]) for g in grads]
        # ^ layers list (batch_size x num output classes x flattened layer weights)
        all_grads = tf.concat(flat_grads, axis=2)
        # ^ batch_size x num output classes x all flattened weights
        two_norms = tf.map_fn(lambda x: tf.norm(x, axis=0), all_grads)
        max_two_norm = tf.reduce_max(two_norms)
        return max_two_norm, y_pred

    def train_step(self, data):
        x, y = data

        with tf.device(self.labels_party_dev):
            if self.disable_encryption:
                enc_y = y
            else:
                key_path = tempfile.mkdtemp()  # Every trace gets a new key.

                shell_context = self.shell_context_fn()
                secret_key = tf_shell.create_key64(
                    shell_context, key_path + "/secret_key"
                )
                secret_fast_rotation_key = tf_shell.create_fast_rotation_key64(
                    shell_context, secret_key, key_path + "/secret_fast_rotation_key"
                )
                if self.needs_public_rotation_key:
                    public_rotation_key = tf_shell.create_rotation_key64(
                        shell_context, secret_key, key_path + "/public_rotation_key"
                    )
                # Encrypt the batch of secret labels y.
                enc_y = tf_shell.to_encrypted(y, secret_key, shell_context)

        with tf.device(self.features_party_dev):
            # Forward pass in plaintext.
            # y_pred = self(x, training=True)
            max_two_norm, y_pred = self.compute_max_two_norm_and_pred(x)

            # Backward pass.
            dx = self.loss_fn.grad(enc_y, y_pred)
            dJ_dw = []
            dJ_dx = [dx]
            for l in reversed(self.layers):
                if isinstance(l, tf_shell_ml.GlobalAveragePooling1D):
                    dw, dx = l.backward(dJ_dx[-1])
                else:
                    dw, dx = l.backward(
                        dJ_dx[-1],
                        (
                            public_rotation_key
                            if not self.disable_encryption
                            and self.needs_public_rotation_key
                            else None
                        ),
                    )
                dJ_dw.extend(dw)
                dJ_dx.append(dx)

            if len(dJ_dw) == 0:
                raise ValueError("No gradients found.")

            # Mask the encrypted grads to prepare for decryption. The masks may
            # overflow during the reduce_sum over the batch. When the masks are
            # operated on, they are multiplied by the scaling factor, so it is
            # not necessary to mask the full range -t/2 to t/2. (Though it is
            # possible, it unnecessarily introduces noise into the ciphertext.)
            if self.disable_masking or self.disable_encryption:
                masked_enc_grads = [g for g in reversed(dJ_dw)]
            else:
                t = tf.cast(shell_context.plaintext_modulus, tf.float32)
                t_half = t // 2
                mask_scaling_factors = [g._scaling_factor for g in reversed(dJ_dw)]
                mask = [
                    tf.random.uniform(
                        tf_shell.shape(g),
                        dtype=tf.float32,
                        minval=-t_half / s,
                        maxval=t_half / s,
                    )
                    for g, s in zip(reversed(dJ_dw), mask_scaling_factors)
                    # tf.zeros_like(tf_shell.shape(g), dtype=tf.int64)
                    # for g in dJ_dw
                ]

                # Mask the encrypted gradients and reverse the order to match
                # the order of the layers.
                masked_enc_grads = [(g + m) for g, m in zip(reversed(dJ_dw), mask)]

        with tf.device(self.labels_party_dev):
            if self.disable_encryption:
                # Unpacking is not necessary when not using encryption.
                masked_grads = masked_enc_grads
            else:
                # Decrypt the weight gradients.
                packed_masked_grads = [
                    tf_shell.to_tensorflow(
                        g,
                        secret_fast_rotation_key if g._is_fast_rotated else secret_key,
                    )
                    for g in masked_enc_grads
                ]

                # Unpack the plaintext gradients using the corresponding layer's
                # unpack function.
                # TODO: Make sure this doesn't require sending the layers
                # themselves just for unpacking. The weights should not be
                # shared with the labels party.
                masked_grads = [
                    f(g) for f, g in zip(self.unpacking_funcs, packed_masked_grads)
                ]

        with tf.device(self.features_party_dev):
            if self.disable_masking or self.disable_encryption:
                grads = masked_grads
            else:
                # SHELL represents floats as integers between [0, t) where t is the
                # plaintext modulus. To mimic the modulo operation without SHELL,
                # numbers which exceed the range [-t/2, t/2) are shifted back into
                # the range.
                epsilon = tf.constant(1e-6, dtype=float)

                def rebalance(x, s):
                    r_bound = t_half / s + epsilon
                    l_bound = -t_half / s - epsilon
                    t_over_s = t / s
                    x = tf.where(x > r_bound, x - t_over_s, x)
                    x = tf.where(x < l_bound, x + t_over_s, x)
                    return x

                # Unmask the gradients using the mask. The unpacking function may
                # sum the mask from two of the gradients (one from each batch), so
                # the mask must be brought back into the range of [-t/2, t/2] before
                # subtracting it from the gradient, and again after.
                unpacked_mask = [f(m) for f, m in zip(self.unpacking_funcs, mask)]
                unpacked_mask = [
                    rebalance(m, s) for m, s in zip(unpacked_mask, mask_scaling_factors)
                ]
                grads = [mg - m for mg, m in zip(masked_grads, unpacked_mask)]
                grads = [rebalance(g, s) for g, s in zip(grads, mask_scaling_factors)]

            # TODO: set stddev based on clipping threshold.
            if self.disable_noise:
                noised_grads = grads
            else:
                noise = [
                    tf.random.normal(
                        tf.shape(g),
                        stddev=max_two_norm * self.noise_multiplier,
                        dtype=float,
                    )
                    # tf.zeros(tf.shape(g))
                    for g in grads
                ]
                noised_grads = [g + n for g, n in zip(grads, noise)]

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(noised_grads, self.weights))

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
            metric_results = {"num_slots": shell_context.num_slots}

        return metric_results

    def test_step(self, data):
        x, y = data

        # Forward pass.
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss.
        self.compute_loss(y=y, y_pred=y_pred)

        # Update the other metrics.
        for metric in self.val_metrics:
            if metric.name != "loss" and metric.name != "num_slots":
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.val_metrics}
