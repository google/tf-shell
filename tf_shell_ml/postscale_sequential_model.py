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
        for l in layers:
            if "tf_shell_ml" in getattr(l, "__module__", None):
                raise ValueError(
                    "tf_shell_ml.PostScaleSequential does not support tf_shell layers"
                )

        super().__init__(layers, *args, **kwargs)

    def call(self, inputs, training=False, with_softmax=True):
        prediction = super().call(inputs, training)
        if not with_softmax:
            return prediction
        # Perform the last layer activation since it is removed for training
        # purposes.
        return tf.nn.softmax(prediction)

    def _predict_and_jacobian(self, features):
        """
        Predicts the output for the given features and optionally computes the
        Jacobian.

        Args:
            features (tf.Tensor): Input features for the model.

        Returns:
            tuple: A tuple containing:
                - predictions (tf.Tensor): The model output after applying
                  softmax.
                - jacobians (list or tf.Tensor): The Jacobian of the last layer
                  preactivation with respect to the model weights.
        """
        with tf.GradientTape(
            persistent=tf.executing_eagerly() or self.jacobian_pfor
        ) as tape:
            predictions = self(features, training=True, with_softmax=False)

        jacobians = tape.jacobian(
            predictions,
            self.trainable_variables,
            # unconnected_gradients=tf.UnconnectedGradients.ZERO, broken with pfor
            parallel_iterations=self.jacobian_pfor_iterations,
            experimental_use_pfor=self.jacobian_pfor,
        )
        # ^  layers list x (batch size x num output classes x weights) matrix
        # dy_pred_j/dW_sample_class

        # Compute the last layer's activation manually since we skipped it above.
        predictions = tf.nn.softmax(predictions)

        return predictions, jacobians

    def _backward(self, dJ_dz, jacobians):
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
            dJ_dz_exp = dJ_dz
            for _ in range(num_weight_dims):
                dJ_dz_exp = tf_shell.expand_dims(dJ_dz_exp, axis=-1)

            # Scale the jacobian.
            scaled_grad = dJ_dz_exp * j
            # ^ batch_size x num output classes x weights

            # Sum over the output classes. At this point, this is a gradient
            # and no longer a jacobian.
            scaled_grad = tf_shell.reduce_sum(scaled_grad, axis=1)
            # ^  batch_size x weights

            grads.append(scaled_grad)

        return grads

    def compute_grads(self, features, enc_labels):
        scaling_factor = (
            enc_labels.scaling_factor
            if hasattr(enc_labels, "scaling_factor")
            else float("inf")
        )
        scaling_factor = tf.cast(scaling_factor, dtype=tf.keras.backend.floatx())

        predictions_list = []
        jacobians_list = []
        max_two_norms_list = []

        with tf.device(self.features_party_dev):
            split_features, end_pad = self.split_with_padding(
                features, len(self.jacobian_devices)
            )

        for i, d in enumerate(self.jacobian_devices):
            with tf.device(d):
                f = tf.identity(split_features[i])  # copy to GPU if needed
                prediction, jacobians = self._predict_and_jacobian(f)
                if i == len(self.jacobian_devices) - 1 and end_pad > 0:
                    # The last device's features may have been padded for even
                    # split jacobian computation across multiple devices.
                    prediction = prediction[:-end_pad]
                    jacobians = jacobians[:-end_pad]
                predictions_list.append(prediction)
                jacobians_list.append(jacobians)

                # Perform PostScale (dJ/dz * dz/dw) in plaintext for every
                # possible label using the worst casse quantization of the
                # jacobian.
                with tf.name_scope("sensitivity_analysis"):
                    sensitivity = tf.constant(0.0, dtype=tf.keras.backend.floatx())
                    worst_case_jacobians = [
                        tf_shell.worst_case_rounding(j, scaling_factor)
                        for j in jacobians
                    ]
                    worst_case_prediction = tf_shell.worst_case_rounding(
                        prediction, scaling_factor
                    )

                    def cond(possible_label_i, sensitivity):
                        return possible_label_i < self.out_classes

                    def body(possible_label_i, sensitivity):
                        possible_label = tf.one_hot(
                            possible_label_i,
                            self.out_classes,
                            dtype=tf.keras.backend.floatx(),
                        )
                        dJ_dz = worst_case_prediction - possible_label
                        possible_grads = self._backward(dJ_dz, worst_case_jacobians)

                        max_norm = self.max_per_example_global_norm(possible_grads)
                        sensitivity = tf.maximum(sensitivity, max_norm)
                        return possible_label_i + 1, sensitivity

                    # Using a tf.while_loop (vs. a python for loop) is preferred
                    # as it does not encode the unrolled loop into the graph and
                    # also allows explicit control over the loop's parallelism.
                    # Increasing parallel_iterations may be faster at the expense
                    # of memory usage.
                    possible_label_i = tf.constant(0)
                    sensitivity = tf.while_loop(
                        cond,
                        body,
                        [possible_label_i, sensitivity],
                        parallel_iterations=1,
                    )[1]

                    max_two_norms_list.append(sensitivity)

        with tf.device(self.features_party_dev):
            predictions = tf.concat(predictions_list, axis=0)
            max_two_norm = tf.reduce_max(max_two_norms_list)
            jacobians = [tf.concat(j, axis=0) for j in zip(*jacobians_list)]

            # Compute prediction - labels (where labels may be encrypted).
            enc_dJ_dz = enc_labels.__rsub__(predictions)  # dJ/dprediction
            # ^  batch_size x num output classes.

            grads = self._backward(enc_dJ_dz, jacobians)

        return grads, max_two_norm, predictions
