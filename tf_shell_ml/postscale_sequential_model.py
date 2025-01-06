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
from tf_shell_ml import large_tensor


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


    def compute_grads(self, features, enc_labels):
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
            scalars = enc_labels.__rsub__(predictions)  # dJ/dprediction
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

        return grads, max_two_norm, predictions
