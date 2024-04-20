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
import numpy as np
import tf_shell


def relu(x):
    return tf.nn.relu(x)


def relu_deriv(y, dy):
    assert not isinstance(y, tf_shell.ShellTensor64)
    # Cannot operate on individual slots of a shell tensor.
    # Formulate the problem as element-wise multiplication.
    mask = tf.where(
        y > 0,
        tf.constant(1, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
    )
    return dy * mask


def sigmoid(x):
    return tf.math.sigmoid(x)


def sigmoid_deriv(y, d_y):
    return d_y * (y * (1 - y))
