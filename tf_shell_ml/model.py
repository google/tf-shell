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


class TfShellSequential(keras.Sequential):
    def __init__(
        self,
        layers,
        shell_context_fn,
        use_encryption,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        *args,
        **kwargs
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
        self.mpc_bit_width = 16

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

    def train_step(self, data):
        x, y = data

        filtered_layers = [l for l in self.layers if hasattr(l, "unpack")]
        unpacking_funcs = [l.unpack for l in filtered_layers]

        with tf.device(self.labels_party_dev):
            if self.use_encryption:
                # Generate the shell context and secret keys. TensorFlow's graph
                # compiler ensures these are only generated once and stored in
                # the graph vs. being regenerated on each call.
                shell_context = self.shell_context_fn()
                secret_key = tf_shell.create_key64(shell_context)
                secret_fast_rotation_key = tf_shell.create_fast_rotation_key64(
                    shell_context, secret_key
                )
                public_rotation_key = tf_shell.create_rotation_key64(
                    shell_context, secret_key
                )
                # Encrypt the batch of secret labels y.
                enc_y = tf_shell.to_encrypted(y, secret_key, shell_context)
            else:
                enc_y = y
                public_rotation_key = None

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
                    dw, dx = l.backward(dJ_dx[-1], public_rotation_key)
                dJ_dw.extend(dw)
                dJ_dx.append(dx)

            # Mask the encrypted grads to prepare for decryption.
            mask = [
                tf.random.uniform(
                    tf_shell.shape(g)[1:],
                    dtype=tf.int64,
                    minval=0,
                    maxval=2**self.mpc_bit_width,
                )
                for g in dJ_dw
            ]
            # Mask the encrypted gradients and reverse the order to match the
            # order of the layers.
            masked_enc_grads = [
                (g + m) for g, m in zip(reversed(dJ_dw), reversed(mask))
            ]

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
                    f(g) for f, g in zip(unpacking_funcs, packed_masked_grads)
                ]
            else:
                masked_grads = masked_enc_grads

            # Flatten the masked gradients from a list of tensors for each layer
            # to a single tensor.
            grad_shapes = [tf.shape(mg) for mg in masked_grads]
            flattened_grad_shapes = [tf.reduce_prod(s) for s in grad_shapes]

            masked_grads = [tf.reshape(mg, [-1]) for mg in masked_grads]
            masked_grads = tf.concat(masked_grads, axis=0)

            # Sample the noise for differential privacy.
            # TODO: set stddev based on clipping threshold.
            noise = tf.random.normal(tf.shape(masked_grads), stddev=1)

            # After decryption, the mask has dtype float. Encode it back to int
            # with shells scaling factor for use in the clip and noise protocol.
            # Do the same for the noise.
            masked_grads = tf.cast(
                tf.round(masked_grads * shell_context.scaling_factor), tf.int64
            )
            noise = tf.cast(tf.round(noise * shell_context.scaling_factor), tf.int64)

            # If running features party and labels party on the same node,
            # skip the MPC protocol.
            # if self.labels_party_dev != self.features_party_dev:
            #     # Start labels party MPC protocol.
            #     tf_shell.clip_and_noise_labels_party(
            #         masked_grads,
            #         self.clipping_threshold,
            #         noise,
            #         Bitwidth=self.mpc_bit_width,
            #         StartPort=5555,
            #         FeaturePartyHost="127.0.0.1",
            #     )

        with tf.device(self.features_party_dev):
            # Encode the mask with the scaling factor for use in the clip and
            # noise protocol.
            mask = [tf.reshape(m, [-1]) for m in mask]
            mask = tf.concat(mask, axis=0)
            # mask = tf.cast(tf.round(mask * shell_context.scaling_factor, tf.int64)

            # If running features party and labels party on the same node,
            # skip the MPC protocol and clip and noise the gradients directly.
            # if self.labels_party_dev != self.features_party_dev:
            #     clipped_noised_grads = tf_shell.clip_and_noise_features_party(
            #         mask,
            #         Bitwidth=self.mpc_bit_width,
            #         StartPort=5555,
            #         LabelPartyHost="127.0.0.1",
            #     )
            # else:
            unmasked_grads = masked_grads - mask
            # clipped_noised_grads = tf.cond(
            #     tf.reduce_sum(unmasked_grads * unmasked_grads)
            #     > self.clipping_threshold,
            #     lambda: self.clipping_threshold + noise,
            #     lambda: unmasked_grads + noise,
            # )
            # clipped_noised_grads = unmasked_grads + noise
            clipped_noised_grads = unmasked_grads

            # Emulate overflow of 2's complement addition between `Bitwidth`
            # integers from when grad + noise is computed under the MPC
            # protocol. Note any overflow in the masking / unmasking cancels
            # out.
            # min_val = -(2 ** (self.mpc_bit_width - 1))
            # max_val = 2 ** (self.mpc_bit_width - 1) - 1
            # clipped_noised_grads = tf.where(
            #     clipped_noised_grads > max_val,
            #     min_val + (clipped_noised_grads - max_val),
            #     clipped_noised_grads,
            # )
            # end else

            # Decode the clipped and noised gradients.
            clipped_noised_grads = (
                tf.cast(clipped_noised_grads, float) / shell_context.scaling_factor
            )

            # Reover the original shapes of the inputs
            clipped_noised_grads = tf.split(clipped_noised_grads, flattened_grad_shapes)
            clipped_noised_grads = [
                tf.reshape(g, s) for g, s in zip(clipped_noised_grads, grad_shapes)
            ]

            weights = []
            for l in filtered_layers:
                weights += l.weights

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(clipped_noised_grads, weights))

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

        # Add the number of slots to the metric results for auto-parameter
        # optimizer.
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

    @tf.function
    def train_step_func(self, data):
        return self.train_step(data)

    def set_dataset_batching(self, train_dataset):
        # Run the training loop once on dummy data to figure out the batch size.
        tf.config.run_functions_eagerly(False)
        metrics = self.train_step_func(next(iter(train_dataset)))
        train_dataset = train_dataset.rebatch(
            metrics["num_slots"].numpy(), drop_remainder=True
        )
        return train_dataset
