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
import tempfile
from tf_shell_ml.model_base import SequentialBase


class PostScaleSequential(SequentialBase):
    def __init__(
        self,
        layers,
        shell_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
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
        self.clipping_threshold = 10000000
        self.context_prepped = False
        self.disable_encryption = disable_encryption
        self.noise_multiplier = noise_multiplier
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

    def _unpack(self, x_list):
        batch_size = tf.shape(x_list[0])[0] // 2
        return [x[0] + x[batch_size] for x in x_list]

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
                # Encrypt the batch of secret labels y.
                enc_y = tf_shell.to_encrypted(y, secret_key, shell_context)

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

            # Remember the original shape of the gradient in order to unpack them
            # after the multiplication so they can be applied to the model.
            # Get the shapes from TensorFlow's tensors, not SHELL's context
            # for when the batch size != slotting dim or not using encryption.
            if len(grads) == 0:
                raise ValueError("No gradients found")
            slot_size = tf.shape(grads[0])[0]
            num_output_classes = grads[0].shape[1]
            grad_shapes = [g.shape[2:] for g in grads]
            flattened_grad_shapes = [s.num_elements() for s in grad_shapes]

            flat_grads = [
                tf.reshape(g, [slot_size, num_output_classes, s])
                for g, s in zip(grads, flattened_grad_shapes)
            ]
            # ^ layers list (batch_size x num output classes x flattened layer weights)

            all_grads = tf.concat(flat_grads, axis=2)
            # ^ batch_size x num output classes x all flattened weights

            # The DP sensitivity of backprop when revealing the sum of gradients
            # across a batch, is the maximum L2 norm of the gradient over every
            # example in the batch, and over every class.
            two_norms = tf.map_fn(lambda x: tf.norm(x, axis=0), all_grads)
            # ^ batch_size x num output classes
            max_two_norm = tf.reduce_max(two_norms)
            # ^ scalar

            scaled_grads = scalars * all_grads
            # ^ batch_size x num output classes x all flattened weights

            # Sum over the output classes.
            scaled_grads = tf_shell.reduce_sum(scaled_grads, axis=1)

            # Split to recover the gradients by layer.
            ps_grads = tf_shell.split(scaled_grads, flattened_grad_shapes, axis=1)
            # ^ layers list (batch_size x flat layer weights)

            # Unflatten the gradients to the original layer shape.
            ps_grads = [
                tf_shell.reshape(
                    g,
                    tf.concat([[slot_size], tf.cast(s, dtype=tf.int64)], axis=0),
                )
                for g, s in zip(ps_grads, grad_shapes)
            ]

            # Mask the encrypted grads to prepare for decryption. The masks may
            # overflow during the reduce_sum over the batch. When the masks are
            # operated on, they are multiplied by the scaling factor, so it is
            # not necessary to mask the full range -t/2 to t/2. (Though it is
            # possible, it unnecessarily introduces noise into the ciphertext.)
            if self.disable_masking or self.disable_encryption:
                masked_enc_grads = ps_grads
            else:
                t = tf.cast(shell_context.plaintext_modulus, tf.float32)
                t_half = t // 2
                mask_scaling_factors = [g._scaling_factor for g in ps_grads]
                masks = [
                    tf.random.uniform(
                        tf_shell.shape(g),
                        dtype=tf.float32,
                        minval=-t_half / s,
                        maxval=t_half / s,
                    )
                    for g, s in zip(ps_grads, mask_scaling_factors)
                ]

                # Mask the encrypted gradients to prepare for decryption.
                masked_enc_grads = [g + m for g, m in zip(ps_grads, masks)]

        with tf.device(self.labels_party_dev):
            if self.disable_encryption:
                masked_grads = masked_enc_grads
            else:
                # Decrypt the weight gradients.
                masked_grads = [
                    tf_shell.to_tensorflow(g, secret_key) for g in masked_enc_grads
                ]

            # Sum the masked gradients over the batch.
            if self.disable_masking or self.disable_encryption:
                batch_masked_grads = [tf.reduce_sum(mg, 0) for mg in masked_grads]
            else:
                batch_masked_grads = [
                    tf_shell.reduce_sum_with_mod(mg, 0, shell_context, s)
                    for mg, s in zip(masked_grads, mask_scaling_factors)
                ]

        with tf.device(self.features_party_dev):
            if self.disable_masking or self.disable_encryption:
                batch_grad = batch_masked_grads
            else:
                # Sum the masks over the batch.
                batch_masks = [
                    tf_shell.reduce_sum_with_mod(m, 0, shell_context, s)
                    for m, s in zip(masks, mask_scaling_factors)
                ]

                # Unmask the batch gradient.
                batch_grad = [mg - m for mg, m in zip(batch_masked_grads, batch_masks)]

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

                batch_grad = [
                    rebalance(g, s) for g, s in zip(batch_grad, mask_scaling_factors)
                ]

            if not self.disable_encryption and self.check_overflow_close:
                # If the unmasked gradient is between [-t/2, -t/4] or
                # [t/4, t/2], the gradient may have overflowed.
                overflowed = [tf.abs(g) > t_half / 2 for g in batch_grad]
                overflowed = [tf.reduce_any(o) for o in overflowed]
                overflowed = tf.reduce_any(overflowed)
                tf.cond(
                    overflowed,
                    lambda: tf.print("Gradient may have overflowed"),
                    lambda: tf.identity(overflowed),
                )

            if self.disable_noise:
                noised_grads = batch_grad
            else:
                # Set the noise based on the maximum two norm of the gradient
                # per example, per output.
                noise = [
                    tf.random.normal(
                        tf.shape(g),
                        stddev=max_two_norm * self.noise_multiplier,
                        dtype=float,
                    )
                    for g in batch_grad
                ]
                # ^ layers list (batch_size x weights)

                noised_grads = [g + n for g, n in zip(batch_grad, noise)]

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
