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
        use_encryption,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        needs_public_rotation_key=False,
        noise_multiplier=1.0,
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
        self.use_encryption = use_encryption
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.clipping_threshold = 10000000
        self.needs_public_rotation_key = needs_public_rotation_key

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

    def train_step(self, data):
        x, y = data

        with tf.device(self.labels_party_dev):
            if self.use_encryption:
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
            else:
                enc_y = y

        with tf.device(self.features_party_dev):
            # Forward pass in plaintext.
            y_pred = self(x, training=True)

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
                        public_rotation_key if self.needs_public_rotation_key else None,
                    )
                dJ_dw.extend(dw)
                dJ_dx.append(dx)

            if len(dJ_dw) == 0:
                raise ValueError("No gradients found.")

            # Setup parameters for the masking.
            t = tf.cast(shell_context.plaintext_modulus, tf.float32)
            t_half = t // 2
            mask_scaling_factors = [g._scaling_factor for g in reversed(dJ_dw)]

            # Mask the encrypted grads to prepare for decryption.
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

            # Mask the encrypted gradients and reverse the order to match the
            # order of the layers.
            masked_enc_grads = [(g + m) for g, m in zip(reversed(dJ_dw), mask)]

        with tf.device(self.labels_party_dev):
            if self.use_encryption:
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
            else:
                masked_grads = masked_enc_grads

        with tf.device(self.features_party_dev):
            # SHELL represents floats as integers between [0, t) where t is the
            # plaintext modulus. To mimic the modulo operation without SHELL,
            # numbers which exceed the range [-t/2, t/2) are shifted back into
            # the range.
            def rebalance(x_list, t_half, scaling_factor_list):
                x_list = [
                    tf.where(x > t_half / s + (1 / s - 1e-6), x - t / s, x)
                    for x, s in zip(x_list, scaling_factor_list)
                ]
                x_list = [
                    tf.where(x < -t_half / s - (1 / s - 1e-6), x + t / s, x)
                    for x, s in zip(x_list, scaling_factor_list)
                ]
                return x_list

            # Unmask the gradients using the mask. The unpacking function may
            # sum the mask from two of the gradients (one from each batch), so
            # the mask must be brought back into the range of [-t/2, t/2] before
            # subtracting it from the gradient, and again after.
            unpacked_mask = [f(m) for f, m in zip(self.unpacking_funcs, mask)]
            unpacked_mask = rebalance(unpacked_mask, t_half, mask_scaling_factors)
            grads = [mg - m for mg, m in zip(masked_grads, unpacked_mask)]
            grads = rebalance(grads, t_half, mask_scaling_factors)

            # TODO: set stddev based on clipping threshold.
            noise = [
                # tf.random.normal(tf.shape(g), stddev=1, dtype=float) for g in grads
                tf.zeros(tf.shape(g))
                for g in grads
            ]
            noised_grads = [g + n for g, n in zip(grads, noise)]

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(noised_grads, self.weights))

        # Do not update metrics during secure training.
        if not self.use_encryption:
            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    loss = self.loss_fn(y, y_pred)
                    metric.update_state(loss)
                else:
                    metric.update_state(y, y_pred)

            metric_results = {m.name: m.result() for m in self.metrics}
        else:
            metric_results = {}

        metric_results["num_slots"] = shell_context.num_slots
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
