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
        use_encryption,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        needs_public_rotation_key=False,
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
        self.mask_bit_width = 16
        self.context_prepped = False
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

    def _unpack(self, x_list):
        batch_size = tf.shape(x_list[0])[0] // 2
        return [x[0] + x[batch_size] for x in x_list]

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
            # Unset the activation function for the last layer so it is not used in
            # computing the gradient. The effect of the last layer activation function
            # is factored out of the gradient computation and accounted for below.
            self.layers[-1].activation = tf.keras.activations.linear

            with tf.GradientTape() as tape:
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

            # Scale each gradient. Since 'scalars' may be a vector of ciphertexts, this
            # requires multiplying plaintext gradient for the specific layer (2d) by the
            # ciphertext (scalar). To do so efficiently under encryption requires
            # flattening and packing the weights, as shown below.
            ps_grads = []
            for layer_grad_full in grads:
                # Remember the original shape of the gradient in order to unpack them
                # after the multiplication so they can be applied to the model.
                batch_sz = layer_grad_full.shape[0]
                num_output_classes = layer_grad_full.shape[1]
                grad_shape = layer_grad_full.shape[2:]

                packable_grad = tf.reshape(
                    layer_grad_full, [batch_sz, num_output_classes, -1]
                )
                # ^  batch_size x num output classes x flattened weights

                # Scale the gradient precursors.
                scaled_grad = scalars * packable_grad
                # ^  dJ/dW = dJ/dy_pred * dy_pred/dW

                # Sum over the output classes.
                scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=1)
                # ^  batch_size x flattened weights

                # In the real world, this approach would also likely require clipping
                # the gradient, and adding DP noise.

                # Reshape to unflatten the weights.
                scaled_grad = tf_shell.reshape(scaled_grad, [batch_sz] + grad_shape)
                # ^  batch_size x weights

                # Sum over the batch.
                if self.use_encryption:
                    if self.needs_public_rotation_key:
                        scaled_grad = tf_shell.reduce_sum(
                            scaled_grad, axis=0, rotation_key=public_rotation_key
                        )
                    else:
                        scaled_grad = tf_shell.fast_reduce_sum(scaled_grad)
                else:
                    scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=0)
                # ^  batch_size x flattened weights

                ps_grads.append(scaled_grad)

            # Mask the encrypted grads to prepare for decryption.
            mask = [
                # tf.random.uniform(
                #     tf_shell.shape(g),
                #     dtype=tf.int64,
                #     minval=0,
                #     maxval=2**self.mask_bit_width,
                # )
                tf.zeros_like(tf_shell.shape(g), dtype=tf.int64)
                for g in ps_grads
            ]

            # Mask the encrypted gradients and reverse the order to match the
            # order of the layers.
            masked_enc_grads = [g + m for g, m in zip(ps_grads, mask)]

        with tf.device(self.labels_party_dev):
            if self.use_encryption:
                # Decrypt the weight gradients.
                packed_masked_grads = [
                    tf_shell.to_tensorflow(
                        g,
                        secret_fast_rotation_key if g._is_fast_rotated else secret_key,
                    )
                    for g in ps_grads
                ]

                # Unpack the plaintext gradients using the corresponding layer's
                # unpack function.
                masked_grads = self._unpack(packed_masked_grads)
            else:
                masked_grads = ps_grads

        with tf.device(self.features_party_dev):
            # Unmask the gradients using the mask.
            # unpacked_mask = self._unpack(mask)
            # unpacked_mask = [tf.cast(m, dtype=float)[0] for m in zip(unpacked_mask)]
            # grads = [mg - m for mg, m in zip(masked_grads, unpacked_mask)]

            # # TODO: set stddev based on clipping threshold.
            # # noise = [
            # #     tf.random.normal(tf.shape(g), stddev=1, dtype=float) for g in grads
            # # ]
            # noise = [
            #     tf.zeros_like(g, dtype=float) for g in grads
            # ]
            # noised_grads = [g + n for g, n in zip(grads, noise)]

            # # Apply the gradients to the model.
            # self.optimizer.apply_gradients(zip(noised_grads, self.weights))
            self.optimizer.apply_gradients(zip(masked_grads, self.weights))

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
