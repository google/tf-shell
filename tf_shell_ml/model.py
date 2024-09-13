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
        labels_party,
        features_party,
        use_encryption,
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
        self.labels_party = labels_party
        self.features_party = features_party
        self.use_encryption = use_encryption

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

        if self.use_encryption:
            # TODO: with device scope of labels party.
            # Generate the shell context and secret keys.
            # TensorFlow's graph compiler ensures these are only generated once
            # and stored in the graph vs. being regenerated on each call.
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

        # Forward pass always in plaintext
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

        enc_grads = reversed(dJ_dw)

        filtered_layers = [l for l in self.layers if len(l.weights) > 0]

        if self.use_encryption:
            # Decrypt the weight gradients. In practice, the gradients should be
            # noised before decrypting.
            if isinstance(l, tf_shell_ml.ShellDense) and l.use_fast_reduce_sum:
                if hasattr(l, "use_fast_reduce_sum") and l.use_fast_reduce_sum:
                    decryption_key = secret_fast_rotation_key
                else:
                    decryption_key = secret_key

                packed_grads = [
                    tf_shell.to_tensorflow(g, decryption_key) for g in enc_grads
                ]
            else:
                packed_grads = [
                    tf_shell.to_tensorflow(g, secret_key) for g in enc_grads
                ]

            # Unpack the plaintext gradients using the corresponding layer.
            grads = [l.unpack(g) for l, g in zip(filtered_layers, packed_grads)]
        else:
            grads = enc_grads

        weights = []
        for l in filtered_layers:
            weights += l.weights

        # Apply the gradients to the model.
        self.optimizer.apply_gradients(zip(grads, weights))

        # Do not update metrics during secure training.
        if not self.use_encryption:
            # Update metrics (includes the metric that tracks the loss)
            for metric in self.metrics:
                if metric.name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(y, y_pred)

        metric_results = {m.name: m.result() for m in self.metrics}
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

    def fit(self, train_dataset, **kwargs):
        # Run the training loop once on dummy data to figure out the batch size.
        tf.config.run_functions_eagerly(False)
        metrics = self.train_step_func(next(iter(train_dataset)))
        train_dataset = train_dataset.rebatch(
            metrics["num_slots"].numpy(), drop_remainder=True
        )

        return super().fit(train_dataset, **kwargs)
