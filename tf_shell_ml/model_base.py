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
from tensorflow.keras.callbacks import CallbackList
import tf_shell
import time
import gc
from tf_shell_ml import large_tensor


class SequentialBase(keras.Sequential):

    def __init__(
        self,
        layers,
        backprop_context_fn,
        noise_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        noise_multiplier=1.0,
        noise_max_scale=500.0,
        noise_base_scale=7.6,
        cache_path=None,
        jacobian_pfor=False,
        jacobian_pfor_iterations=None,
        jacobian_devices=None,
        disable_encryption=False,
        disable_masking=False,
        disable_noise=False,
        check_overflow_INSECURE=False,
        *args,
        **kwargs,
    ):
        super().__init__(layers, *args, **kwargs)
        self.backprop_context_fn = backprop_context_fn
        self.noise_context_fn = noise_context_fn
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.noise_multiplier = noise_multiplier
        self.cache_path = cache_path
        self.jacobian_pfor = jacobian_pfor
        self.jacobian_pfor_iterations = jacobian_pfor_iterations
        self.jacobian_devices = (
            [features_party_dev] if jacobian_devices is None else jacobian_devices
        )
        self.disable_encryption = disable_encryption
        self.disable_masking = disable_masking
        self.disable_noise = disable_noise
        self.check_overflow_INSECURE = check_overflow_INSECURE
        self.dataset_prepped = False

        self.dg_params = tf_shell.DiscreteGaussianParams(
            max_scale=noise_max_scale, base_scale=noise_base_scale
        )

        if self.disable_encryption and self.jacobian_pfor:
            print(
                "WARNING: `jacobian_pfor` may be incompatible with `disable_encryption`."
            )

        if len(self.layers) > 0 and not (
            self.layers[-1].activation is tf.keras.activations.softmax
            or self.layers[-1].activation is tf.nn.softmax
        ):
            raise ValueError(
                "The model must have a softmax activation function on the final layer. Saw",
                self.layers[-1].activation,
            )

        if len(self.jacobian_devices) == 0:
            raise ValueError("No devices specified for Jacobian computation.")

    def compile(self, loss, **kwargs):
        if not isinstance(loss, tf.keras.losses.CategoricalCrossentropy):
            raise ValueError(
                "The loss function must be tf.keras.losses.CategoricalCrossentropy. Saw",
                loss,
            )

        super().compile(
            jit_compile=False,  # Disable XLA, no CPU op for tf_shell_ml's TensorArrayV2.
            loss=loss,
            **kwargs,
        )

    def train_step(self, features, labels):
        metrics, num_slots = self.shell_train_step(
            features, labels, read_key_from_cache=False
        )
        return metrics

    @tf.function
    def train_step_with_keygen(self, features, labels):
        return self.shell_train_step(features, labels, read_key_from_cache=False)

    @tf.function
    def train_step_tf_func(self, features, labels):
        # When running the training loop, the caches for encryption keys and
        # contexts have already been populated and do not need to be read.
        return self.shell_train_step(features, labels, read_key_from_cache=True)

    def compute_grads(self, features, enc_labels):
        raise NotImplementedError()  # Should be overloaded by the subclass.

    def prep_dataset_for_model(self, train_features, train_labels):
        """Prepare the dataset for training with encryption by setting the batch
        size to the same value as the encryption ring degree. Run the training
        loop once on dummy data to figure out the batch size.
        """
        if self.disable_encryption:
            self.batch_size = next(iter(train_features)).shape[0]
            self.dataset_prepped = True
            return train_features, train_labels

        # Run the training loop once on dummy data to figure out the batch size.
        # Use a separate tf.function to avoid caching the trace so keys and
        # context are written to cache and read on next trace.
        metrics, num_slots = self.train_step_with_keygen(
            next(iter(train_features)), next(iter(train_labels))
        )

        self.batch_size = num_slots.numpy()

        with tf.device(self.features_party_dev):
            train_features = train_features.rebatch(
                num_slots.numpy(), drop_remainder=True
            )
        with tf.device(self.labels_party_dev):
            train_labels = train_labels.rebatch(num_slots.numpy(), drop_remainder=True)

        self.dataset_prepped = True
        return train_features, train_labels

    def fast_prep_dataset_for_model(self, train_features, train_labels):
        """Prepare the dataset for training with encryption by setting the
        batch size to the same value as the encryption ring degree. It is faster
        than `prep_dataset_for_model` because it does not execute the graph,
        instead tracing and optimizing the graph and extracting the required
        parameters without actually executing the graph.

        Since the graph is not executed, caches for keys and the shell context
        are not written to disk.
        """
        if self.disable_encryption:
            self.batch_size = next(iter(train_features)).shape[0]
            self.dataset_prepped = True
            return train_features, train_labels

        # Call the training step with keygen to trace the graph. Note, since the
        # graph is not executed, the caches for encryption keys and contexts are
        # not written to disk.
        func = self.train_step_with_keygen.get_concrete_function(
            next(iter(train_features)), next(iter(train_labels))
        )

        # Optimize the graph using tf_shells HE-specific optimizers.
        optimized_func = tf_shell.optimize_shell_graph(
            func, skip_convert_to_constants=True
        )
        optimized_graph = optimized_func.graph

        def find_node_by_op(g, name):
            for node in g.as_graph_def().node:
                if node.op == name:
                    return node
            raise ValueError(f"Node {name} not found in graph.")

        # Using parameters in the optimized graph, create the context and
        # keys for use during training. The parameters are pulled from the
        # graph because if autocontext is used, these parameters are not
        # known until the graph optimization pass is finished.
        context_node = find_node_by_op(optimized_graph, "ContextImport64")

        def get_tensor_by_name(g, name):
            for node in g.as_graph_def().node:
                if node.name == name:
                    return tf.make_ndarray(node.attr["value"].tensor)
            raise ValueError(f"Node {name} not found in graph.")

        log_n = get_tensor_by_name(optimized_graph, context_node.input[0]).tolist()
        self.batch_size = 2**log_n

        with tf.device(self.features_party_dev):
            train_features = train_features.rebatch(2**log_n, drop_remainder=True)
        with tf.device(self.labels_party_dev):
            train_labels = train_labels.rebatch(2**log_n, drop_remainder=True)

        self.dataset_prepped = True
        return train_features, train_labels

    def fit(
        self,
        features_dataset,
        labels_dataset,
        epochs=1,
        batch_size=32,
        callbacks=None,
        validation_data=None,
        steps_per_epoch=None,
        verbose=1,
    ):
        """A custom training loop that supports inputs from multiple datasets,
        each of which can be on a different device.
        """
        # Prevent TensorFlow from placing ops on devices which were not
        # explicitly assigned for security reasons.
        tf.config.set_soft_device_placement(False)

        # Turn on the shell optimizers.
        tf_shell.enable_optimization()

        if not self.dataset_prepped:
            features_dataset, labels_dataset = self.prep_dataset_for_model(
                features_dataset, labels_dataset
            )
            tf.keras.backend.clear_session()
            gc.collect()

        # Calculate samples if possible.
        if steps_per_epoch is None:
            samples = None
        else:
            samples = steps_per_epoch * self.batch_size

        # Initialize callbacks.
        callback_list = CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            batch_size=self.batch_size,
            epochs=epochs,
            steps=steps_per_epoch,
            samples=samples,
            verbose=verbose,
            do_validation=validation_data is not None,
            metrics=list(self.metrics_names),
        )

        # Begin training.
        callback_list.on_train_begin()
        logs = {}

        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch, logs)
            start_time = time.time()
            self.reset_metrics()

            # Training loop.
            for step, (batch_x, batch_y) in enumerate(
                zip(features_dataset, labels_dataset)
            ):
                callback_list.on_train_batch_begin(step, logs)
                logs, num_slots = self.train_step_tf_func(batch_x, batch_y)
                callback_list.on_train_batch_end(step, logs)
                gc.collect()
                if steps_per_epoch is not None and step + 1 >= steps_per_epoch:
                    break

            # Validation loop.
            if validation_data is not None:
                # Reset metrics
                self.reset_metrics()

                for val_x_batch, val_y_batch in validation_data:
                    val_y_pred = self(val_x_batch, training=False)
                    # Update validation metrics
                    for m in self.metrics:
                        if m.name == "loss":
                            loss = self.compiled_loss(val_y_batch, val_y_pred)
                            m.update_state(loss)
                        else:
                            m.update_state(val_y_batch, val_y_pred)
                metric_results = {m.name: m.result() for m in self.metrics}

                # TensorFlow 2.18.0 added a "CompiledMetrics" metric which holds
                # metrics passed to compile in it's own dictionary. Keras wants
                # all metrics to be returned as a flat dictionary. Here we
                # flatten the dictionary.
                result = {}
                for key, value in metric_results.items():
                    if isinstance(value, dict):
                        result.update(value)  # add subdict directly into the dict
                    else:
                        result[key] = value  # non-subdict elements are just copied

                logs.update({f"val_{name}": result for name, result in result.items()})

                # End of epoch.
                logs["time"] = time.time() - start_time

                # Update the steps in callback parameters with actual steps completed
                if steps_per_epoch is None:
                    steps_per_epoch = step + 1
                    samples = steps_per_epoch * self.batch_size
                    callback_list.params["steps"] = steps_per_epoch
                    callback_list.params["samples"] = samples
                callback_list.on_epoch_end(epoch, logs)

        # End of training.
        callback_list.on_train_end(logs)
        return self.history

    def split_with_padding(self, tensor, num_splits, axis=0, padding_value=0):
        """Splits a tensor along the given axis, padding if necessary."""

        # Pad the tensor if necessary
        remainder = tensor.shape[axis] % num_splits
        padded_by = 0
        if remainder != 0:
            padded_by = num_splits - remainder
            padding = [[0, 0] for _ in range(tensor.shape.rank)]
            padding[axis][1] = padded_by
            tensor = tf.pad(tensor, padding, constant_values=padding_value)

        # Split the tensor
        return tf.split(tensor, num_splits, axis=axis), padded_by

    def predict_and_jacobian(self, features, skip_jacobian=False):
        with tf.GradientTape(
            persistent=tf.executing_eagerly() or self.jacobian_pfor
        ) as tape:
            predictions = self(features, training=True, with_softmax=False)

        if skip_jacobian:
            jacobians = []
        else:
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

    def jacobian_max_two_norm(self, jacobians):
        """Takes the output of the jacobian computation and computes the max two
        norm of the weights over all examples in the batch and all output
        classes. Do this layer-wise to reduce memory usage."""
        if len(jacobians) == 0:
            return tf.constant(0.0, dtype=tf.keras.backend.floatx())

        batch_size = jacobians[0].shape[0]
        num_output_classes = jacobians[0].shape[1]
        sum_of_squares = tf.zeros(
            [batch_size, num_output_classes], dtype=tf.keras.backend.floatx()
        )

        for j in jacobians:
            # Ignore the batch size and num output classes dimensions and
            # recover just the number of dimensions in the weights.
            num_weight_dims = len(j.shape) - 2
            reduce_sum_dims = range(2, 2 + num_weight_dims)
            sum_of_squares += tf.reduce_sum(j * j, axis=reduce_sum_dims)

        return tf.sqrt(tf.reduce_max(sum_of_squares))

    def flatten_and_pad_grad_list(self, grads_list, slot_size):
        """Takes as input a list of tensors and flattens them into a single
        tensor. The input is expected to be of shape:
            layers list x (weights)
        where weights may be any shape. The output is a tensor of shape:
            slot_size x remaining flattened weights
        which is the input weights flattened, concatenated, and padded out to
        make the output shape non-ragged.
        """
        if len(grads_list) == 0:
            raise ValueError("No gradients found")

        grad_shapes = [g.shape for g in grads_list]
        flattened_grad_shapes = [s.num_elements() for s in grad_shapes]
        total_grad_size = sum(flattened_grad_shapes)

        flat_grad_list = [tf.reshape(g, [-1]) for g in grads_list]
        # ^ layers list x (flattened weights)

        flat_grads = tf.concat(flat_grad_list, axis=0)
        # ^ all flattened weights

        pad_len = slot_size - tf.math.floormod(
            tf.cast(total_grad_size, dtype=tf.int64), slot_size
        )
        padded_flat_grads = tf.concat(
            [flat_grads, tf.zeros(pad_len, dtype=flat_grads.dtype)], axis=0
        )
        out = tf.reshape(padded_flat_grads, [slot_size, -1])

        return out, grad_shapes, flattened_grad_shapes, total_grad_size

    def unflatten_and_unpad_grad(
        self, flat_grads, grad_shapes, flattened_grad_shapes, total_grad_size
    ):
        """Takes as input a flattened and padded gradient tensor and unflattens
        it into a list of tensors. This undoes the flattening and padding
        introduced by flatten_and_pad_grad_list(). The input is expected to be
        of shape:
            slot_size x remaining flattened weights
        The output is a list of tensors of shape:
            layers list x (weights)
        """

        # First reshape to a flat tensor.
        flat_grads = tf.reshape(flat_grads, [-1])

        # Remove the padding.
        flat_grads = flat_grads[:total_grad_size]

        # Split the flat tensor into the original shapes.
        grads_list = tf_shell.split(flat_grads, flattened_grad_shapes, axis=0)

        # Reshape to the original shapes.
        grads_list = [tf.reshape(g, s) for g, s in zip(grads_list, grad_shapes)]
        return grads_list

    def mask_gradients(self, context, grads):
        """Adds a random masks to the encrypted gradients between [-t/2, t/2]."""
        if self.disable_masking or self.disable_encryption:
            return grads, None

        t = tf.cast(tf.identity(context.plaintext_modulus), dtype=tf.int64)
        t_half = t // 2
        mask_scaling_factors = [g._scaling_factor for g in grads]

        masks = [
            tf.random.uniform(
                tf_shell.shape(g),
                dtype=tf.int64,
                minval=-t_half,
                maxval=t_half,
            )
            for g in grads
        ]

        # Spoof the gradient's dtype to int64 and scaling factor to 1.
        # This is necessary so when adding the masks to the gradients, tf_shell
        # does not attempt to do any casting to floats matching scaling factors.
        # Conversion back to floating point and handling the scaling factors is
        # done in unmask_gradients.
        int64_grads = [
            tf_shell.ShellTensor64(
                _raw_tensor=g._raw_tensor,
                _context=g._context,
                _level=g._level,
                _num_mod_reductions=g._num_mod_reductions,
                _underlying_dtype=tf.int64,  # Spoof the dtype.
                _scaling_factor=1,  # Spoof the scaling factor.
                _is_enc=g._is_enc,
                _is_fast_rotated=g._is_fast_rotated,
            )
            for g in grads
        ]

        # Add the masks.
        int64_grads = [(g + m) for g, m in zip(int64_grads, masks)]

        return int64_grads, masks, mask_scaling_factors

    def unmask_gradients(self, context, grads, masks, mask_scaling_factors):
        """Subtracts the masks from the gradients."""

        # Sum the masks over the batch.
        sum_masks = [
            tf_shell.reduce_sum_with_mod(m, 0, context, s)
            for m, s in zip(masks, mask_scaling_factors)
        ]

        # Unmask the batch gradient.
        masks_and_grads = [tf.stack([-m, g]) for m, g in zip(sum_masks, grads)]
        unmasked_grads = [
            tf_shell.reduce_sum_with_mod(mg, 0, context, s)
            for mg, s in zip(masks_and_grads, mask_scaling_factors)
        ]

        # Recover the floating point values using the scaling factors.
        unmasked_grads = [
            tf.cast(g, tf.keras.backend.floatx()) / s
            for g, s in zip(unmasked_grads, mask_scaling_factors)
        ]

        return unmasked_grads

    def compute_noise_factors(self, context, secret_key, sensitivity):
        bounded_sensitivity = tf.cast(sensitivity, dtype=tf.float32) * tf.sqrt(2.0)

        tf.assert_less(
            bounded_sensitivity,
            self.dg_params.max_scale,
            message="Sensitivity is too large for the maximum noise scale.",
        )

        a, b = tf_shell.sample_centered_gaussian_f(bounded_sensitivity, self.dg_params)

        def _prep_noise_factor(x):
            x = tf.expand_dims(x, 0)
            x = tf.expand_dims(x, 0)
            x = tf.repeat(x, context.num_slots, axis=0)
            return tf_shell.to_encrypted(x, secret_key, context)

        enc_a = _prep_noise_factor(a)
        enc_b = _prep_noise_factor(b)
        return enc_a, enc_b

    def noise_gradients(self, context, flat_grads, enc_a, enc_b):
        def _sample_noise():
            n = tf_shell.sample_centered_gaussian_l(
                context,
                tf.size(flat_grads, out_type=tf.int64),
                self.dg_params,
            )
            # The shape prefix of the noise samples must match the shape
            # of the masked gradients. The last dimension is the noise
            # sub-samples.
            n = tf.reshape(
                n, tf.concat([tf.shape(flat_grads), [tf.shape(n)[1]]], axis=0)
            )
            return n

        y1 = _sample_noise()
        y2 = _sample_noise()

        enc_x1 = tf_shell.reduce_sum(enc_a * y1, axis=2)
        enc_x2 = tf_shell.reduce_sum(enc_b * y2, axis=2)
        enc_noise = enc_x1 + enc_x2
        grads = enc_noise + flat_grads
        return grads

    def warn_on_overflow(self, grads, scaling_factors, plaintext_modulus, message):
        """If the gradient is between [-t/2, -t/4] or [t/4, t/2], the gradient
        may have overflowed. This also must take the scaling factor into account
        so the range is divided by the scaling factor.
        """
        # t = tf.cast(plaintext_modulus, grads[0].dtype)
        t_half = plaintext_modulus / 2

        over_by = [
            tf.reduce_max(tf.abs(g) - tf.cast(t_half / 2 / s, g.dtype))
            for g, s in zip(grads, scaling_factors)
        ]
        max_over_by = tf.reduce_max(over_by)
        overflowed = tf.reduce_any(max_over_by > 0)

        tf.cond(
            overflowed,
            lambda: tf.print(
                message,
                "Overflowed by",
                over_by,
                "(positive number indicates overflow amount).",
                "Values should be less than",
                [tf.cast(t_half / 2 / s, grads[0].dtype) for s in scaling_factors],
            ),
            lambda: tf.identity(overflowed),
        )

        return overflowed


    def shell_train_step(self, features, labels, read_key_from_cache=False):
        """
        The main training loop for the PostScale protocol.

        Args:
            backprop_context: ShellContext for backpropagation, containing
                encryption parameters.
            backprop_secret_key: Secret key for decrypting backpropagated
                gradients.
            read_key_from_cache: Boolean whether to read keys from cache.
            predictions: Model predictions, plaintext.
            grads: Gradients to be masked and noised.
            max_two_norm: Maximum L2 norm to determine noise scale.
            labels: Real labels corresponding to the predictions.

        Returns:
            result: Dictionary of metric results.
            num_slots: Number of slots in the backpropagation context, or None
                if encryption is disabled.
        """
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_encryption:
                enc_labels = labels
            else:
                backprop_context = self.backprop_context_fn(read_key_from_cache)
                backprop_secret_key = tf_shell.create_key64(
                    backprop_context, self.cache_path
                )
                # Encrypt the batch of secret labels.
                enc_labels = tf_shell.to_encrypted(
                    labels, backprop_secret_key, backprop_context
                )

        # Call the derived class to compute the gradients.
        grads, max_two_norm, predictions = self.compute_grads(features, enc_labels)

        with tf.device(self.features_party_dev):
            # Check if the backproped gradients overflowed.
            if not self.disable_encryption and self.check_overflow_INSECURE:
                # Note, checking the gradients requires decryption on the
                # features party which breaks security of the protocol.
                bp_scaling_factors = [g._scaling_factor for g in grads]
                dec_grads = [
                    tf_shell.to_tensorflow(g, backprop_secret_key) for g in grads
                ]
                self.warn_on_overflow(
                    dec_grads,
                    bp_scaling_factors,
                    tf.identity(backprop_context.plaintext_modulus),
                    "WARNING: Backprop gradient may have overflowed.",
                )

            # Mask the encrypted gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads, masks, mask_scaling_factors = self.mask_gradients(
                    backprop_context, grads
                )

            # Prepare to send gradients to the labels party.
            chunked_grads, chunked_grads_metadata = large_tensor.split_tensor_list(
                grads
            )

            if not self.disable_noise:
                # Set up the features party side of the distributed noise
                # sampling sub-protocol.
                noise_context = self.noise_context_fn(read_key_from_cache)
                noise_secret_key = tf_shell.create_key64(noise_context, self.cache_path)

                # The noise context must have the same number of slots
                # (encryption ring degree) as used in backpropagation.
                if not self.disable_encryption:
                    tf.assert_equal(
                        tf.identity(backprop_context.num_slots),
                        noise_context.num_slots,
                        message="Backprop and noise contexts must have the same number of slots.",
                    )

                # The noise scaling factor must always be 1. Encryptions already
                # have the scaling factor applied when the noise is applied,
                # and an additional noise scaling factor is not needed.
                tf.assert_equal(
                    noise_context.scaling_factor,
                    1,
                    message="Noise scaling factor must be 1.",
                )

                # Compute the noise factors for the distributed noise sampling
                # sub-protocol.
                enc_a, enc_b = self.compute_noise_factors(
                    noise_context, noise_secret_key, max_two_norm
                )

        with tf.device(self.labels_party_dev):
            # Prepare to send gradients to the labels party.
            grads = large_tensor.reassemble_tensor_list(
                chunked_grads, chunked_grads_metadata
            )

            if not self.disable_encryption:
                # Decrypt the weight gradients with the backprop key.
                grads = [tf_shell.to_tensorflow(g, backprop_secret_key) for g in grads]

            # Sum the masked gradients over the batch.
            if self.disable_masking or self.disable_encryption:
                # No mask has been added so a only a normal sum is required.
                grads = [tf.reduce_sum(g, axis=0) for g in grads]
            else:
                grads = [
                    tf_shell.reduce_sum_with_mod(g, 0, backprop_context, s)
                    for g, s in zip(grads, mask_scaling_factors)
                ]

            if not self.disable_noise:
                # Efficiently pack the masked gradients to prepare for adding
                # the encrypted noise. This is special because the masked
                # gradients are no longer batched, so the packing must be done
                # manually.
                (flat_grads, grad_shapes, flattened_grad_shapes, total_grad_size) = (
                    self.flatten_and_pad_grad_list(
                        grads, tf.identity(noise_context.num_slots)
                    )
                )

                # Add the encrypted noise to the masked gradients.
                grads = self.noise_gradients(noise_context, flat_grads, enc_a, enc_b)

        with tf.device(self.features_party_dev):
            if not self.disable_noise:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = tf_shell.to_tensorflow(grads, noise_secret_key)

                # Unpack the noised grads after decryption.
                grads = self.unflatten_and_unpad_grad(
                    flat_grads,
                    grad_shapes,
                    flattened_grad_shapes,
                    total_grad_size,
                )
            else:
                # Ensure the gradients are copied to the correct device in the
                # case when noise is disabled but encryption and masking are
                # both enabled.
                grads = [tf.identity(g) for g in grads]

            # Unmask the gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads = self.unmask_gradients(
                    backprop_context, grads, masks, mask_scaling_factors
                )

            if not self.disable_noise:
                if self.check_overflow_INSECURE:
                    self.warn_on_overflow(
                        grads,
                        [1] * len(grads),
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )

            # Apply the gradients to the model.
            self.optimizer.apply_gradients(zip(grads, self.weights))

            for metric in self.metrics:
                if metric.name == "loss":
                    if self.disable_encryption:
                        loss = self.compiled_loss(labels, predictions)
                        metric.update_state(loss)
                    else:
                        # Loss is unknown when encrypted.
                        metric.update_state(0.0)
                else:
                    if self.disable_encryption:
                        metric.update_state(labels, predictions)
                    else:
                        # Other metrics are uknown when encrypted.
                        zeros = tf.broadcast_to(0, tf.shape(predictions))
                        metric.update_state(zeros, zeros)

            metric_results = {m.name: m.result() for m in self.metrics}

            # TensorFlow 2.18.0 added a "CompiledMetrics" metric which holds metrics
            # passed to compile in it's own dictionary. Keras wants all metrics to
            # be returned as a flat dictionary. Here we flatten the dictionary.
            result = {}
            for key, value in metric_results.items():
                if isinstance(value, dict):
                    result.update(value)  # add subdict directly into the dict
                else:
                    result[key] = value  # non-subdict elements are just copied

            return result, (
                None
                if self.disable_encryption
                else tf.identity(backprop_context.num_slots)
            )
