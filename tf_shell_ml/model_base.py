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


class SequentialBase(keras.Sequential):

    def __init__(
        self,
        layers,
        *args,
        **kwargs,
    ):
        super().__init__(layers, *args, **kwargs)
        self.dataset_prepped = False

    @tf.function
    def train_step_tf_func(self, data):
        return self.train_step(data)

    # Prepare the dataset for training with encryption by setting the batch size
    # to the same value as the encryption ring degree. Run the training loop once
    # on dummy data to figure out the batch size.
    def prep_dataset_for_model(self, train_dataset):
        if self.disable_encryption:
            self.dataset_prepped = True
            return train_dataset

        # Run the training loop once on dummy data to figure out the batch size.
        tf.config.run_functions_eagerly(False)
        metrics = self.train_step_tf_func(next(iter(train_dataset)))

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Expected train_step to return a dict, got {type(metrics)}."
            )

        if "num_slots" not in metrics:
            raise ValueError(
                f"Expected train_step to return a dict with key 'num_slots', got {metrics.keys()}."
            )

        train_dataset = train_dataset.rebatch(
            metrics["num_slots"].numpy(), drop_remainder=True
        )

        self.dataset_prepped = True
        return train_dataset

    # Prepare the dataset for training with encryption by setting the batch size
    # to the same value as the encryption ring degree. It is faster than
    # `prep_dataset_for_model` because it does not execute the graph, instead
    # tracing and optimizing the graph and extracting the required parameters.
    def fast_prep_dataset_for_model(self, train_dataset):
        if not self.disable_encryption:
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

        train_dataset = train_dataset.rebatch(2**log_n, drop_remainder=True)
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
