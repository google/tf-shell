// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ScalarShape;
using tensorflow::shape_inference::ShapeHandle;

// Tensorflow does not have size_t but Shell Context parameters require it.
// Code below must assume size_t is a unit64 because of this.
static_assert(sizeof(size_t) == sizeof(uint64_t), "Fatal size mismatch");

REGISTER_OP("ContextImport64")
    .Input("log_n: uint64")
    .Input("main_moduli: uint64")
    .Input("aux_moduli: uint64")
    .Input("plaintext_modulus: uint64")
    .Input("noise_variance: uint64")
    .Input("seed: string")
    .Output("shell_context: variant")
    .SetIsStateful()
    .SetShapeFn(ScalarShape);

REGISTER_OP("PolynomialImport64")
    .Attr("dtype: {uint8, int8, int16, int32, int64, float, double}")
    .Input("shell_context: variant")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle output;

      // First dimension of "in" is stored via shell Polynomial.
      TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 1, &output));

      c->set_output(0, output);
      return OkStatus();
    });

// Output shape depends on content of context object
// so no SetShapeFn() for this Op.
REGISTER_OP("PolynomialExport64")
    .Attr("dtype: {uint8, int8, int16, int32, int64}")
    .Input("shell_context: variant")
    .Input("in: variant")
    .Output("val: dtype")
    .SetIsStateful();

REGISTER_OP("KeyGen64")
    .Input("context: variant")
    .Output("key: variant")
    .SetIsStateful()
    .SetShapeFn(ScalarShape);

REGISTER_OP("Encrypt64")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Output("out: variant")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return OkStatus();
    });

// Output shape depends on content of shell_context
// so no SetShapeFn() for this Op.
REGISTER_OP("Decrypt64")
    .Attr("dtype: {uint8, int8, int16, int32, int64}")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Output("out: dtype")
    .SetIsStateful();

// Add and subtract.
REGISTER_OP("AddCtCt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("AddCtPt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("AddPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("SubCtCt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("SubCtPt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("SubPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("NegCt64")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetIsStateful();

REGISTER_OP("NegPt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetIsStateful();

// Multiply.
REGISTER_OP("MulCtCt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("MulCtPt64")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("MulPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("MatMulCtPt64")
    .Attr("dtype: {uint8, int8, int16, int32, int64, float, double}")
    .Input("a: variant")
    .Input("b: dtype")
    .Output("c: variant")
    .SetIsStateful();

REGISTER_OP("MatMulPtCt64")
    .Attr("dtype: {uint8, int8, int16, int32, int64, float, double}")
    .Input("a: dtype")
    .Input("b: variant")
    .Output("c: variant")
    .SetIsStateful();