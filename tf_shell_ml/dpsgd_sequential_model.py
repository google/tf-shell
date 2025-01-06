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

    def compute_grads(self, features, enc_labels):
        predictions_list = []
        max_two_norms_list = []
        split_features, end_pad = self.split_with_padding(
            features, len(self.jacobian_devices)
        )
        for i, d in enumerate(self.jacobian_devices):
            with tf.device(d):
                f = tf.identity(split_features[i])  # copy to GPU if needed
                prediction, jacobians = self.predict_and_jacobian(
                    f,
                    skip_jacobian=self.disable_noise,  # Jacobian only needed for noise.
                )
                if i == len(self.jacobian_devices) - 1 and end_pad > 0:
                    # The last device's features may have been padded for even
                    # split jacobian computation across multiple devices.
                    prediction = prediction[:-end_pad]
                    jacobians = jacobians[:-end_pad]
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
            dx = enc_labels.__rsub__(predictions)  # Derivative of CCE loss and softmax.
            dJ_dw = []  # Derivatives of the loss with respect to the weights.
            dJ_dx = [dx]  # Derivatives of the loss with respect to the inputs.
            for l in reversed(self.layers):
                dw, dx = l.backward(dJ_dx[-1])
                dJ_dw.extend(dw)
                dJ_dx.append(dx)

            # Reverse the order to match the order of the layers.
            grads = [g for g in reversed(dJ_dw)]

        return grads, max_two_norm, predictions