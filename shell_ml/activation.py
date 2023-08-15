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
import shell_tensor


def relu(x):
    return tf.nn.relu(x)


def relu_deriv(y, dy):
    if isinstance(dy, shell_tensor.ShellTensor64):
        # Cannot operate on individual slots of a shell tensor.
        # Formulate the problem as element-wise multiplication.
        t = np.dtype(dy.plaintext_dtype.as_numpy_dtype)
        mask = tf.where(y <= 0, t.type(0), t.type(1))
        return dy * mask
    else:
        return dy * (y > 0)


def sigmoid(x):
    return tf.math.sigmoid(x)


def sigmoid_deriv(y, d_y):
    return d_y * (y * (1 - y))
