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
import numpy as np
import tensorflow as tf


class Adam:
    """
    M = b1 * M + (1-b1) * G
    V = b2 * V + (1-b2) * G^2
    M_hat = M / (1 - beta1**t)
    V_hat = V / (1 - beta2**t)
    W = W - r * M_hat / sqrt(V_hat)

    https://paperswithcode.com/method/adam
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=0.00001):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Ms = {}
        self.Vs = {}
        self.beta1_pows = {}
        self.beta2_pows = {}
        self.epsilon = epsilon

    def compile(self, weights):
        for W in weights:
            self.Vs[id(W)] = [
                np.zeros(list(map(int, W[i].shape.as_list()))) for i in range(len(W))
            ]
            self.Ms[id(W)] = [
                np.zeros(list(map(int, W[i].shape.as_list()))) for i in range(len(W))
            ]

            self.beta1_pows[id(W)] = np.array(1, dtype=np.float64)
            self.beta2_pows[id(W)] = np.array(1, dtype=np.float64)

    def grad_to_weight(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Weights not found")

        with tf.name_scope("adam-gradients-to-weights"):
            beta1_pow = self.beta1_pows[id(W)]
            beta2_pow = self.beta2_pows[id(W)]
            beta1_pow *= self.beta1
            beta2_pow *= self.beta2

            M = self.Ms[id(W)]
            V = self.Vs[id(W)]
            for i in range(len(W)):
                M[i] = self.beta1 * M[i] + (1 - self.beta1) * G[i]
                V[i] = self.beta2 * V[i] + (1 - self.beta2) * G[i] * G[i]
                mhat = M[i] / (1 - beta1_pow)
                vhat = V[i] / (1 - beta2_pow)
                diff = self.learning_rate * mhat * tf.math.rsqrt(vhat + self.epsilon)
                W[i] -= diff
