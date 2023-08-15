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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

shell_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_shell_ops.so")
)
context_import64 = shell_ops.context_import64
polynomial_import64 = shell_ops.polynomial_import64
polynomial_export64 = shell_ops.polynomial_export64
prng_import = shell_ops.prng_import
key_gen64 = shell_ops.key_gen64

encrypt64 = shell_ops.encrypt64
decrypt64 = shell_ops.decrypt64

add_ct_ct64 = shell_ops.add_ct_ct64
add_ct_pt64 = shell_ops.add_ct_pt64
add_pt_pt64 = shell_ops.add_pt_pt64

sub_ct_ct64 = shell_ops.sub_ct_ct64
sub_ct_pt64 = shell_ops.sub_ct_pt64
sub_pt_pt64 = shell_ops.sub_pt_pt64

neg_ct64 = shell_ops.neg_ct64
neg_pt64 = shell_ops.neg_pt64

mul_ct_ct64 = shell_ops.mul_ct_ct64
mul_ct_pt64 = shell_ops.mul_ct_pt64
mul_pt_pt64 = shell_ops.mul_pt_pt64
mat_mul_ct_pt64 = shell_ops.mat_mul_ct_pt64
mat_mul_pt_ct64 = shell_ops.mat_mul_pt_ct64
