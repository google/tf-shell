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


class SequentialBase(keras.Sequential):

    def __init__(
        self,
        layers,
        backprop_context_fn,
        noise_context_fn,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        noise_multiplier=1.0,
        cache_path=None,
        jacobian_pfor=False,
        jacobian_pfor_iterations=None,
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
        self.disable_encryption = disable_encryption
        self.disable_masking = disable_masking
        self.disable_noise = disable_noise
        self.check_overflow_INSECURE = check_overflow_INSECURE
        self.dataset_prepped = False

        if self.disable_encryption and self.jacobian_pfor:
            print(
                "WARNING: `jacobian_pfor` may be incompatible with `disable_encryption`."
            )

    def compile(self, shell_loss, **kwargs):
        if not isinstance(shell_loss, tf_shell_ml.CategoricalCrossentropy):
            raise ValueError(
                "The model must be used with the tf-shell version of CategoricalCrossentropy loss function. Saw",
                shell_loss,
            )
        if len(self.layers) > 0 and not (
            self.layers[-1].activation is tf.keras.activations.softmax
            or self.layers[-1].activation is tf.nn.softmax
        ):
            raise ValueError(
                "The model must have a softmax activation function on the final layer. Saw",
                self.layers[-1].activation,
            )

        if shell_loss is None:
            raise ValueError("shell_loss must be provided")
        self.loss_fn = shell_loss

        super().compile(loss=tf.keras.losses.CategoricalCrossentropy(), **kwargs)

    def train_step(self, data):
        metrics, num_slots = self.shell_train_step(data)
        return metrics

    @tf.function
    def train_step_tf_func(self, data):
        return self.shell_train_step(data)

    def shell_train_step(self, data):
        raise NotImplementedError()  # Should be overloaded by the subclass.

    # Prepare the dataset for training with encryption by setting the batch size
    # to the same value as the encryption ring degree. Run the training loop once
    # on dummy data to figure out the batch size.
    def prep_dataset_for_model(self, train_dataset):
        if self.disable_encryption:
            self.dataset_prepped = True
            return train_dataset

        # Run the training loop once on dummy data to figure out the batch size.
        tf.config.run_functions_eagerly(False)
        metrics, num_slots = self.train_step_tf_func(next(iter(train_dataset)))

        train_dataset = train_dataset.rebatch(num_slots.numpy(), drop_remainder=True)

        self.dataset_prepped = True
        return train_dataset

    # Prepare the dataset for training with encryption by setting the batch size
    # to the same value as the encryption ring degree. It is faster than
    # `prep_dataset_for_model` because it does not execute the graph, instead
    # tracing and optimizing the graph and extracting the required parameters
    # without actually executing the graph.
    def fast_prep_dataset_for_model(self, train_dataset):
        if self.disable_encryption:
            self.dataset_prepped = True
            return train_dataset

        # Call the training step with keygen to trace the graph. Use a copy
        # of the function to avoid caching the trace.
        traceable_copy = self.train_step_tf_func
        func = traceable_copy.get_concrete_function(next(iter(train_dataset)))

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

        train_dataset = train_dataset.unbatch().batch(2**log_n, drop_remainder=True)
        self.dataset_prepped = True
        return train_dataset

    def fit(self, train_dataset, **kwargs):
        # Prevent TensorFlow from placing ops on devices which were not
        # explicitly assigned for security reasons.
        tf.config.set_soft_device_placement(False)

        # Turn on the shell optimizers.
        tf_shell.enable_optimization()

        if not self.dataset_prepped:
            train_dataset = self.prep_dataset_for_model(train_dataset)

        return super().fit(train_dataset, **kwargs)

    def flatten_jacobian_list(self, grads):
        """Takes as input a jacobian and flattens into a single tensor. The
        jacobian is expected to be of the form:
            layers list x (batch size x num output classes x weights)
        where weights may be any shape. The output is a tensor of shape:
            batch_size x num output classes x all flattened weights
        where all flattened weights include weights from all the layers.
        """
        if len(grads) == 0:
            raise ValueError("No gradients found")

        # Get the shapes from TensorFlow's tensors, not SHELL's context for when
        # the batch size != slotting dim or not using encryption.
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

        return all_grads, slot_size, grad_shapes, flattened_grad_shapes

    def flat_jacobian_two_norm(self, flat_jacobian):
        """Computes the maximum L2 norm of a flattened jacobian. The input is
        expected to be of shape:
            batch_size x num output classes x all flattened weights
        where all flattened weights includes weights from all the layers."""
        # The DP sensitivity of backprop when revealing the sum of gradients
        # across a batch, is the maximum L2 norm of the gradient over every
        # example in the batch, and over every class.
        two_norms = tf.map_fn(lambda x: tf.norm(x, axis=0), flat_jacobian)
        # ^ batch_size x num output classes
        max_two_norm = tf.reduce_max(two_norms)
        # ^ scalar
        return max_two_norm

    def unflatten_batch_grad_list(
        self, flat_grads, slot_size, grad_shapes, flattened_grad_shapes
    ):
        """Takes as input a flattened gradient tensor and unflattens it into a
        list of tensors. This is useful to undo the flattening performed by
        flat_jacobian_list() after the output class dimension has been reduced.
        The input is expected to be of shape:
            batch_size x all flattened weights
        where all flattened weights includes weights from all the layers. The
        output is a list of tensors of shape:
            layers list x (batch size x weights)
        """
        # Split to recover the gradients by layer.
        grad_list = tf_shell.split(flat_grads, flattened_grad_shapes, axis=1)
        # ^ layers list (batch_size x flattened weights)

        # Unflatten the gradients to the original layer shape.
        grad_list = [
            tf_shell.reshape(
                g,
                tf.concat([[slot_size], tf.cast(s, dtype=tf.int64)], axis=0),
            )
            for g, s in zip(grad_list, grad_shapes)
        ]
        # ^ layers list (batch_size x weights)
        return grad_list

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
        padded_flat_grads = tf.concat([flat_grads, tf.zeros(pad_len)], axis=0)
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

    def warn_on_overflow(self, grads, scaling_factors, plaintext_modulus, message):
        # If the gradient is between [-t/2, -t/4] or [t/4, t/2], the gradient
        # may have overflowed. This also must take the scaling factor into
        # account so the range is divided by the scaling factor.
        t = tf.cast(plaintext_modulus, grads[0].dtype)
        t_half = t / 2

        over_by = [
            tf.reduce_max(tf.abs(g) - t_half / 2 / s)
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
                [t_half / 2 / s for s in scaling_factors],
            ),
            lambda: tf.identity(overflowed),
        )

        return overflowed
