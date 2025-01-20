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
import tf_shell_ml


class SequentialBase(keras.Sequential):

    def __init__(
        self,
        layers,
        backprop_context_fn,
        noise_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        noise_multiplier=1.0,
        noise_max_scale=5.0e9,
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
        self.noise_multiplier = float(noise_multiplier)
        self.cache_path = cache_path
        self.jacobian_pfor = jacobian_pfor
        self.jacobian_pfor_iterations = jacobian_pfor_iterations
        self.jacobian_devices = (
            [features_party_dev] if jacobian_devices is None else jacobian_devices
        )
        self.disable_encryption = disable_encryption
        self.disable_masking = disable_masking
        self.disable_noise = False if noise_multiplier == 0.0 else disable_noise
        self.check_overflow_INSECURE = check_overflow_INSECURE
        self.dataset_prepped = False

        self.dg_params = tf_shell.DiscreteGaussianParams(
            max_scale=noise_max_scale, base_scale=noise_base_scale
        )

        if self.disable_encryption and self.jacobian_pfor:
            print(
                "WARNING: `jacobian_pfor` may be incompatible with `disable_encryption`."
            )

        if len(self.layers) > 0:
            if not (
                self.layers[-1].activation is tf.keras.activations.softmax
                or self.layers[-1].activation is tf.nn.softmax
            ):
                raise ValueError(
                    "The model must have a softmax activation function on the final layer. Saw",
                    self.layers[-1].activation,
                )

            self.out_classes = self.layers[-1].units

        # Unset the last layer's activation function. Derived classes handle
        # the softmax activation manually.
        layers[-1].activation = tf.keras.activations.linear

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
        """
        Executes a single training step.

        Args:
            features: The input features for the training step.
            labels: The corresponding labels for the training step.

        Returns:
            metrics (dict): The metrics resulting from the training step.
        """
        # When called from outside this class, the caches for encryption keys
        # and contexts are not guaranteed to be populated, so set
        # read_key_from_cache to False.
        metrics, batch_size_should_be = self.shell_train_step(
            features, labels, read_key_from_cache=False, apply_gradients=True
        )
        return metrics

    @tf.function
    def train_step_tf_func(
        self, features, labels, read_key_from_cache, apply_gradients
    ):
        """
        tf.function wrapper around shell train step.
        """
        return self.shell_train_step(
            features, labels, read_key_from_cache, apply_gradients
        )

    def compute_grads(self, features, enc_labels):
        raise NotImplementedError()  # Should be overloaded by the subclass.

    def prep_dataset_for_model(self, train_features, train_labels):
        """
        Prepare the dataset for training with encryption.

        Sets the batch size to the same value as the encryption ring degree. Run
        the training loop once on dummy inputs to figure out the batch size.
        Note the batch size is not always known during graph construction when
        using autocontext, only after graph optimization.

        Args:
            train_features (tf.data.Dataset): The training features dataset.
            train_labels (tf.data.Dataset): The training labels dataset.

        Returns:
            tuple: A tuple containing the re-batched training features and labels.

        """
        # If neither encryption nor noise are enabled, no encryption is used
        # and the batch size is the same as the input batch size.
        if self.disable_encryption and self.disable_noise:
            self.batch_size = next(iter(train_features)).shape[0]
            self.dataset_prepped = True
            return train_features, train_labels

        # Run the training loop once on dummy data. This writes encryption
        # contexts and keys to caches.
        metrics, batch_size_should_be = self.train_step_tf_func(
            next(iter(train_features)),
            next(iter(train_labels)),
            read_key_from_cache=False,
            apply_gradients=False,
        )

        self.batch_size = batch_size_should_be.numpy()

        with tf.device(self.features_party_dev):
            train_features = train_features.rebatch(
                self.batch_size, drop_remainder=True
            )
        with tf.device(self.labels_party_dev):
            train_labels = train_labels.rebatch(self.batch_size, drop_remainder=True)

        self.dataset_prepped = True
        return train_features, train_labels

    def fast_prep_dataset_for_model(self, train_features, train_labels):
        """
        Prepare the dataset for training with encryption.

        Sets the batch size of the training datasets to the same value as the
        encryption ring degree. This method is faster than
        `prep_dataset_for_model` because it does not execute the graph. Instead,
        it traces and optimizes the graph, extracting the required parameters
        without actually executing it.

        Args:
            train_features (tf.data.Dataset): The training features dataset.
            train_labels (tf.data.Dataset): The training labels dataset.

        Returns:
            tuple: A tuple containing the re-batched training features and labels.
        """
        if self.disable_encryption and self.disable_noise:
            self.batch_size = next(iter(train_features)).shape[0]
            self.dataset_prepped = True
            return train_features, train_labels

        # Call the training step with keygen to trace the graph. Note, since the
        # graph is not executed, the caches for encryption keys and contexts are
        # not written to disk.
        func = self.train_step_tf_func.get_concrete_function(
            next(iter(train_features)),
            next(iter(train_labels)),
            read_key_from_cache=False,
            apply_gradients=False,
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
        """
        Train the model.

        This custom training loop supports inputs from multiple datasets
        (features and laabels), each of which can be located on a different
        device.

        Args:
            features_dataset (tf.data.Dataset): The dataset containing the
                features for training.
            labels_dataset (tf.data.Dataset): The dataset containing the labels
                for training.
            epochs (int, optional): Number of epochs to train the model.
                Defaults to 1.
            batch_size (int, optional): Number of samples per gradient update.
                Defaults to 32.
            callbacks (list, optional): List of `keras.callbacks.Callback`
                instances. Defaults to None.
            validation_data (tuple, optional): Data on which to evaluate the
                loss and any model metrics at the end of each epoch. Defaults to
                None.
            steps_per_epoch (int, optional): Total number of steps (batches of
                samples) before declaring one epoch finished and starting the
                next epoch. Defaults to None.
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress
                bar. Defaults to 1.

        Returns:
            History: A keras `History` object. Its `History.history` attribute
                is a record of training loss values and metrics values at
                successive epochs, as well as validation loss values and
                validation metrics values (if applicable).
        """
        # Prevent TensorFlow from placing ops on devices which were not
        # explicitly assigned for security reasons.
        tf.config.set_soft_device_placement(False)

        # Turn on the shell optimizers.
        tf_shell.enable_optimization()

        # Enable randomized rounding.
        tf_shell.enable_randomized_rounding()

        if not self.dataset_prepped:
            features_dataset, labels_dataset = self.fast_prep_dataset_for_model(
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
        subsequent_run = False

        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch, logs)
            start_time = time.time()
            self.reset_metrics()

            # Training loop.
            for step, (batch_x, batch_y) in enumerate(
                zip(features_dataset, labels_dataset)
            ):
                callback_list.on_train_batch_begin(step, logs)
                # The caches for encryption keys and contexts have already been
                # populated during the dataset preparation step. Set
                # read_key_from_cache to True.
                logs, batch_size_should_be = self.train_step_tf_func(
                    batch_x,
                    batch_y,
                    read_key_from_cache=subsequent_run,
                    apply_gradients=True,
                )
                subsequent_run = True
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
        """
        Splits a tensor along the given axis, padding if necessary.

        Args:
            tensor (tf.Tensor): The input tensor to be split. Requires the shape
                of the tensor is known.
            num_splits (int): The number of splits to divide the tensor into.
            axis (int, optional): The axis along which to split the tensor.
                Defaults to 0.
            padding_value (int, optional): The value to use for padding if the
                tensor's size along the specified axis is not divisible by
                num_splits. Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - List[tf.Tensor]: A list of tensors resulting from the split.
                - int: The remainder of the division of the tensor's size along
                  the specified axis by num_splits.
        """
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

    def flatten_and_pad_grad(self, grad, dim_0_sz):
        num_el = tf.cast(tf.size(grad), dtype=tf.int64)

        flat = tf.reshape(grad, [-1])
        pad_len = dim_0_sz - tf.math.floormod(num_el, dim_0_sz)
        padded = tf.concat([flat, tf.zeros(pad_len, dtype=grad.dtype)], axis=0)
        flat_padded = tf.reshape(padded, [dim_0_sz, -1])
        metadata = {
            "original_shape": tf.shape(grad),
            "padding": pad_len,
        }
        return flat_padded, metadata

    def unflatten_grad(self, grad, metadata):
        padding = metadata["padding"]
        orig_shape = metadata["original_shape"]

        flat_grad = tf.reshape(grad, [-1])
        unpadded_flat_grad = flat_grad[:-padding]
        shaped_grad = tf.reshape(unpadded_flat_grad, orig_shape)
        return shaped_grad

    def flatten_and_pad_grad_list(self, grads_list, dim_0_sz):
        all_grads = []
        all_metadata = []
        for grad in grads_list:
            flat, metadata = self.flatten_and_pad_grad(grad, dim_0_sz)
            all_grads.append(flat)
            all_metadata.append(metadata)

        return all_grads, all_metadata

    def unflatten_grad_list(self, flat_grads, metadata_list):
        grads = []
        for flat, metadata in zip(flat_grads, metadata_list):
            grad = self.unflatten_grad(flat, metadata)
            grads.append(grad)

        return grads

    def max_per_example_global_norm(self, grads):
        """
        Computes the maximum per-example global norm of the gradients.

        Args:
            grads (list of tf.Tensor): The list of gradient tensors.

        Returns:
            tf.Tensor: Scalar of the maximum global norm of the gradients.
        """
        if len(grads) == 0:
            return tf.constant(0.0, dtype=tf.keras.backend.floatx())

        # Sum of squares accumulates the l2 norm squared of the per-example
        # gradients.
        sum_of_squares = tf.zeros([grads[0].shape[0]], dtype=tf.keras.backend.floatx())
        for g in grads:
            # Ignore the batch size and num output classes
            # dimensions and recover just the number of dimensions
            # in the weights.
            num_weight_dims = len(g.shape) - 1
            weight_dims = range(1, 1 + num_weight_dims)
            sum_of_squares += tf.reduce_sum(tf.square(g), axis=weight_dims)

        max_norm = tf.sqrt(tf.reduce_max(sum_of_squares))
        return max_norm

    def spoof_int_gradients(self, grads):
        """
        Spoofs the gradients to have dtype int64 and a scaling factor of 1.

        Args:
            grads (list of tf_shell.ShellTensor): The list of gradient tensors
                to be spoofed.

        Returns:
            list of tf_shell.ShellTensor64: The list of gradient tensors with
                dtype int64 and scaling factor 1.
        """
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
        return int64_grads

    def unspoof_int_gradients(self, int64_grads, orig_scaling_factors):
        """
        Recover the floating point gradients from spoofed int64 gradients using
        the original scaling factors.

        Args:
            int64_grads (list of tf.Tensor): List of gradients that have been
                spoofed as int64 with scaling factor 1.
            orig_scaling_factors (list of float): List of original scaling
                factors used to spoof the gradients.

        Returns:
            list of tf.Tensor: List of recovered floating point gradients.
        """
        grads = [
            tf.cast(g, tf.keras.backend.floatx()) / s
            for g, s in zip(int64_grads, orig_scaling_factors)
        ]
        return grads

    def mask_gradients(self, context, grads):
        """
        Adds random masks to the encrypted gradients within the range [-t/2,
        t/2].

        Args:
            context (object): The context containing the plaintext modulus.
            grads (list): A list of gradient tensors to be masked.

        Returns:
        tuple: A tuple containing:
            - int64_grads (list): The masked gradients with spoofed dtype and
              scaling factor.
            - masks (list): The random masks added to the gradients.
            - mask_scaling_factors (list): The original scaling factors of the
              gradients.
        """
        t = tf.cast(tf.identity(context.plaintext_modulus), dtype=tf.int64)
        t_half = t // 2

        masks = [
            tf.random.uniform(
                tf_shell.shape(g),
                dtype=tf.int64,
                minval=-t_half,
                maxval=t_half,
            )
            for g in grads
        ]

        # Convert to shell plaintexts manually to avoid using the scaling factor
        # in the 'grads[i]' contexts when the addition happens in the next line
        # below.
        shell_masks = [
            tf_shell.to_shell_plaintext(m, context, override_scaling_factor=1)
            for m in masks
        ]

        # Add the masks.
        masked_grads = [(g + m) for g, m in zip(grads, shell_masks)]

        return masked_grads, masks

    def unmask_gradients(self, context, grads, masks):
        """
        Unmasks the gradients by subtracting the masks, and converting to
        floating point representation using the scaling factors.

        Args:
            context: The context in which the operation is performed.
            grads (list of tf.Tensor): The gradients to be unmasked.
            masks (list of tf.Tensor): The masks to be subtracted from the
              gradients.
            mask_scaling_factors (list of float): The scaling factors for the
              masks.

        Returns:
            list of tf.Tensor: The unmasked gradients.
        """
        # Sum the masks over the batch.
        sum_masks = [tf_shell.reduce_sum_with_mod(m, 0, context, 1) for m in masks]

        # Unmask the batch gradient.
        masks_and_grads = [tf.stack([-m, g]) for m, g in zip(sum_masks, grads)]
        unmasked_grads = [
            tf_shell.reduce_sum_with_mod(mg, 0, context, 1) for mg in masks_and_grads
        ]

        return unmasked_grads

    def compute_noise_factors(
        self, context, secret_key, max_two_norm, mask_scaling_factors
    ):
        """
        Computes noise selection vectors, as part of the distributed noise
        sampling protocol to sample the discrete gaussian distribution.

        Args:
            context: An object containing the context information, including the
              number of slots for encryption.
            secret_key: The secret key used for encryption.
            max_two_norm: Maximum L2 norm of per-example gradients.
              factors.

        Returns:
            dict: A dictionary containing the following arrays, one per scaling factor:
                - enc_a: The encrypted noise selection vector `a`.
                - enc_b: The encrypted noise selection vector `b`.

        Raises:
            tf.errors.InvalidArgumentError: If the bounded sensitivity exceeds
              the maximum noise scale defined in `self.dg_params`.
        """
        sensitivity = tf.cast(max_two_norm, dtype=tf.float32)
        sensitivity *= self.noise_multiplier

        # Ensure the sensitivity is at least as large as the base scale, since
        # the base_scale is the smallest scale distribution which the protocol
        # can sample.
        bounded_sensitivity = tf.maximum(sensitivity, self.dg_params.base_scale)

        noise_factors = {
            "enc_as": [],
            "enc_bs": [],
        }

        for sf in mask_scaling_factors:
            # When the noise is added to the gradients, the gradients are scaled
            # by the `mask scaling factors`. The noise must be scaled by the
            # same amount.
            scaled_sensitivity = bounded_sensitivity * sf

            # The noise protocol generates samples in the range [-6s, 6s], where
            # s is the scale. Ensure the maximum noise is less than the
            # plaintext modulus.
            tf.assert_less(
                scaled_sensitivity * 6,
                tf.cast(context.plaintext_modulus, tf.float32),
                message="Sensitivity is too large for the noise context's plaintext modulus and WILL overflow. Reduce the sensitivity by reducing the gradient norms (e.g. reducing the backprop context's scaling factor), or increase the noise context's plaintext modulus.",
            )

            tf.assert_less(
                scaled_sensitivity,
                self.dg_params.max_scale,
                message="Sensitivity is too large for the maximum noise scale.",
            )

            a, b = tf_shell.sample_centered_gaussian_f(
                scaled_sensitivity, self.dg_params
            )

            def _prep_noise_factor(x):
                x = tf.expand_dims(x, 0)
                x = tf.expand_dims(x, 0)
                x = tf.repeat(x, context.num_slots, axis=0)
                return tf_shell.to_encrypted(x, secret_key, context)

            noise_factors["enc_as"].append(_prep_noise_factor(a))
            noise_factors["enc_bs"].append(_prep_noise_factor(b))

        return noise_factors

    def noise_gradients(self, context, flat_grads, noise_factors):
        """
        Adds encrypted noise to the gradients for differential privacy, as
        part of the distributed noise sampling protocol.

        Args:
            context: The context in which the noise is sampled.
            flat_grads: The flattened gradients to which noise is added.
            enc_a: Encrypted noise selection vector.
            enc_b: Encrypted noise selection vector.

        Returns:
            Tensor: The noised gradients.
        """

        noised_grads = []
        for grad, enc_a, enc_b in zip(
            flat_grads, noise_factors["enc_as"], noise_factors["enc_bs"]
        ):

            def _sample_noise():
                n = tf_shell.sample_centered_gaussian_l(
                    context,
                    tf.size(grad, out_type=tf.int64),
                    self.dg_params,
                )
                # The shape prefix of the noise samples must match the shape
                # of the masked gradients. The last dimension is the noise
                # sub-samples.
                n = tf.reshape(n, tf.concat([tf.shape(grad), [tf.shape(n)[1]]], axis=0))
                return n

            y1 = _sample_noise()
            y2 = _sample_noise()

            enc_x1 = tf_shell.reduce_sum(enc_a * y1, axis=2)
            enc_x2 = tf_shell.reduce_sum(enc_b * y2, axis=2)
            enc_noise = enc_x1 + enc_x2
            noised_grads.append(enc_noise + grad)

        return noised_grads

    def warn_on_overflow(self, grads, scaling_factors, plaintext_modulus, message):
        """
        Print a warning if any of the gradients indicate overflow.

        This function checks if any of the gradients are within the ranges
        [-t/2, -t/4] or [t/4, t/2], where t is the plaintext modulus, which may
        indicate an overflow. The check takes into account the scaling factors,

        Args:
            grads (list of tf.Tensor): List of gradient tensors.
            scaling_factors (list of float): List of scaling factors
              corresponding to each gradient tensor.
            plaintext_modulus (float): The plaintext modulus value.
            message (str): The message to print if an overflow is detected.
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

    def shell_train_step(self, features, labels, read_key_from_cache, apply_gradients):
        """
        The main training loop for the PostScale protocol.

        Args:
            features (tf.Tensor): Training features.
            labels (tf.Tensor): Training labels.
            read_key_from_cache: Boolean whether to read keys from cache.
            apply_gradients: Boolean whether to apply gradients.

        Returns:
            result: Dictionary of metric results.
            batch_size_should_be: Number of slots in encryption (ciphertext ring
                degree), or None if encryption is disabled.
        """
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_encryption:
                enc_labels = labels
            else:
                backprop_context = self.backprop_context_fn(read_key_from_cache)
                backprop_secret_key = tf_shell.create_key64(
                    backprop_context, read_key_from_cache, self.cache_path
                )
                # Encrypt the batch of secret labels.
                enc_labels = tf_shell.to_encrypted(
                    labels, backprop_secret_key, backprop_context
                )

        # Call the derived class to compute the gradients.
        grads, max_two_norm, predictions = self.compute_grads(features, enc_labels)

        with tf.device(self.features_party_dev):
            # Store the scaling factors of the gradients.
            if self.disable_encryption:
                backprop_scaling_factors = [1] * len(grads)
            else:
                backprop_scaling_factors = [g._scaling_factor for g in grads]

                # Simple correctness test to check if the gradients are smaller
                # than the plaintext modulus.
                tf.assert_less(
                    max_two_norm,
                    tf.cast(
                        backprop_context.plaintext_modulus, tf.keras.backend.floatx()
                    ),
                    message="Gradient may be too large for the backprop context's plaintext modulus. Reduce the sensitivity by reducing the gradient norms (e.g. reducing the backprop scaling factor), or increase the backprop context's plaintext modulus.",
                )

            # Check if the backproped gradients overflowed.
            if not self.disable_encryption and self.check_overflow_INSECURE:
                # Note, checking the gradients requires decryption on the
                # features party which breaks security of the protocol.
                dec_grads = [
                    tf_shell.to_tensorflow(g, backprop_secret_key) for g in grads
                ]
                self.warn_on_overflow(
                    dec_grads,
                    backprop_scaling_factors,
                    tf.identity(backprop_context.plaintext_modulus),
                    "WARNING: Backprop gradient may have overflowed.",
                )

            # Spoof encrypted gradients as int64 with scaling factor 1 for
            # subsequent masking and noising, so the intermediate decryption
            # is not affected by the scaling factors.
            if not self.disable_encryption:
                grads = self.spoof_int_gradients(grads)

            # Mask the encrypted gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads, masks = self.mask_gradients(backprop_context, grads)

            if self.features_party_dev != self.labels_party_dev:
                # When the tensor needs to be sent between machines, split it
                # into chunks to stay within GRPC's 4BG maximum message size.
                # Note, running this on a single machine sometimes breaks
                # TensorFlow's optimizer, which then decides to replace the
                # entire training graph with a const op.
                chunked_grads, chunked_grads_metadata = (
                    tf_shell_ml.large_tensor.split_tensor_list(grads)
                )

            if not self.disable_noise:
                # Set up the features party side of the distributed noise
                # sampling sub-protocol.
                noise_context = self.noise_context_fn(read_key_from_cache)
                noise_secret_key = tf_shell.create_key64(
                    noise_context, read_key_from_cache, self.cache_path
                )

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
                noise_factors = self.compute_noise_factors(
                    noise_context,
                    noise_secret_key,
                    max_two_norm,
                    backprop_scaling_factors,
                )

        with tf.device(self.labels_party_dev):
            if self.features_party_dev != self.labels_party_dev:
                # Reassemble the tensor list after sending it between machines.
                grads = tf_shell_ml.large_tensor.reassemble_tensor_list(
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
                    tf_shell.reduce_sum_with_mod(g, 0, backprop_context, 1)
                    for g in grads
                ]

            if not self.disable_noise:
                # Efficiently pack the masked gradients to prepare for adding
                # the encrypted noise. This is special because the masked
                # gradients are no longer batched, so the packing must be done
                # manually.
                flat_grads, flat_metadata = self.flatten_and_pad_grad_list(
                    grads, tf.identity(noise_context.num_slots)
                )

                # Add the encrypted noise to the masked gradients.
                grads = self.noise_gradients(noise_context, flat_grads, noise_factors)

        with tf.device(self.features_party_dev):
            if not self.disable_noise:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = [
                    tf_shell.to_tensorflow(g, noise_secret_key) for g in grads
                ]

                # Unpack the noised grads after decryption.
                grads = self.unflatten_grad_list(flat_grads, flat_metadata)

            # Unmask the gradients.
            if not self.disable_masking and not self.disable_encryption:
                grads = self.unmask_gradients(backprop_context, grads, masks)

            # Recover the original scaling factor of the gradients if they were
            # originally encrypted.
            if not self.disable_encryption:
                grads = self.unspoof_int_gradients(grads, backprop_scaling_factors)

            if not self.disable_noise:
                if self.check_overflow_INSECURE:
                    self.warn_on_overflow(
                        grads,
                        [1] * len(grads),
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )

            # When encryption is disabled but the noise protocol is enabled,
            # the gradients are ints. Convert them to floats.
            grads = [tf.cast(g, dtype=tf.keras.backend.floatx()) for g in grads]

            # Normalize the gradient by the batch size.
            grads = [g / tf.cast(tf.shape(g)[0], g.dtype) for g in grads]

            # Apply the gradients to the model.
            if apply_gradients:
                self.optimizer.apply_gradients(zip(grads, self.weights))
            else:
                # If the gradients should not be applied, add zeros instead so
                # the optimizer internal variables are created. To ensure the
                # `grads` are not removed by the grappler optimizer, add them to
                # the UPDATE_OPS collection.
                [
                    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, g)
                    for g in grads
                ]
                zeros = [tf.zeros_like(g) for g in grads]
                self.optimizer.apply_gradients(zip(zeros, self.weights))

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

            if not self.disable_encryption:
                ret_batch_size = tf.identity(backprop_context.num_slots)
            elif not self.disable_noise:
                ret_batch_size = tf.identity(noise_context.num_slots)
            else:
                ret_batch_size = None

            return result, ret_batch_size
