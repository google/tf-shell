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
from tensorflow.nn import softmax
from tensorflow.math import log


class CategoricalCrossentropy:
    def __init__(self, from_logits=False, lazy_normalization=True):
        self.from_logits = from_logits
        self.lazy_normalization = lazy_normalization

    def __call__(self, y_true, y_pred):
        if self.from_logits:
            y_pred = softmax(y_pred)
        batch_size = y_true.shape.as_list()[0]
        batch_size_inv = 1 / batch_size
        out = -y_true * log(y_pred)
        cce = out.reduce_sum() * batch_size_inv
        return cce

    def grad(self, y_true, y_pred):
        if self.from_logits:
            y_pred = softmax(y_pred)

        # When using deferred execution, we need to use the __rsub__ method
        # otherwise it tries to go through the y tensors __sub__ method which
        # fails when y_true is encrypted (a ShellTensor64).
        # grad = y_pred - y_true
        grad = y_true.__rsub__(y_pred)

        if not self.lazy_normalization:
            # batch_size = y_true.shape.as_list()[0]
            # batch_size_inv = 1 / batch_size
            # grad = grad * batch_size_inv
            raise NotImplementedError("Multiply by scalar op not implemented.")
        return grad
