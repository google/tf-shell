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
using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::DimensionHandle;
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
    .SetShapeFn(ScalarShape);

REGISTER_OP("PolynomialImport64")
    .Attr(
        "Dtype: {uint8, int8, int16, uint16, int32, uint32, int64, uint64, "
        "float, double}")
    .Input("shell_context: variant")
    .Input("in: Dtype")
    .Output("val: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle output;

      // First dimension of "in" is stored via shell Polynomial.
      TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 1, &output));

      c->set_output(0, output);
      return OkStatus();
    });

REGISTER_OP("PolynomialExport64")
    .Attr("dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Attr("batching_dim: int")
    .Input("shell_context: variant")
    .Input("in: variant")
    .Output("val: dtype")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 batching_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("batching_dim", &batching_dim));
      ShapeHandle batching_dim_shape = c->MakeShape({batching_dim});

      ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->Concatenate(batching_dim_shape, c->input(1), &output));

      c->set_output(0, output);
      return OkStatus();
    });

REGISTER_OP("KeyGen64")
    .Input("context: variant")
    .Output("key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("Encrypt64")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Output("out: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return OkStatus();
    });

REGISTER_OP("Decrypt64")
    .Attr("dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Attr("batching_dim: int")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Output("out: dtype")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 batching_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("batching_dim", &batching_dim));
      ShapeHandle batching_dim_shape = c->MakeShape({batching_dim});

      ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->Concatenate(batching_dim_shape, c->input(2), &output));

      c->set_output(0, output);
      return OkStatus();
    });

// Add and subtract.
REGISTER_OP("AddCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("AddCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("AddPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("SubCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("SubCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("SubPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("NegCt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("NegPt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

// Multiply.
REGISTER_OP("MulCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("MulCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("MulCtTfScalar64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("MulPtTfScalar64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("MulPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto a_sz = c->NumElements(c->input(1));
      auto b_sz = c->NumElements(c->input(2));
      DimensionHandle out_sz;
      TF_RETURN_IF_ERROR(c->Max(a_sz, b_sz, &out_sz));

      if (c->Value(out_sz) == c->Value(a_sz)) {
        c->set_output(0, c->input(1));
      } else if (c->Value(out_sz) == c->Value(b_sz)) {
        c->set_output(0, c->input(2));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });

REGISTER_OP("MatMulCtPt64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      // Output has the same shape as the plaintext b outer dim.
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(2), 1, &output));
      c->set_output(0, output);
      return OkStatus();
    });

REGISTER_OP("MatMulPtCt64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("rotation_key: variant")
    .Input("a: Dtype")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      // Output has the same shape as the plaintext b outer dim.
      tsl::int32 a_rank = c->Rank(c->input(2));
      ShapeHandle a_shape_prefix;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input(2), 0, a_rank - 2, &a_shape_prefix));

      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->Concatenate(a_shape_prefix, c->input(3), &output));

      c->set_output(0, output);
      return OkStatus();
    });

// Rotate.
REGISTER_OP("RotationKeyGen64")
    .Input("context: variant")
    .Input("key: variant")
    .Output("rotation_key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("Roll64")
    .Input("rotation_key: variant")
    .Input("value: variant")
    .Input("shift: int64")
    .Output("rotated_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("ReduceSumByRotation64")
    .Input("value: variant")
    .Input("rotation_key: variant")
    .Output("repeated_reduce_sum: variant")
    .SetShapeFn([](InferenceContext* c) {
      // ReduceSum over the packing dimension does not change the shape.
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_OP("ReduceSum64")
    .Input("value: variant")
    .Attr("axis: int")
    .Output("repeated_reduce_sum: variant")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 rank = c->Rank(c->input(0));

      tsl::int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

      // Check that axis is in the correct range.
      if (axis == 0) {
        return InvalidArgument(
            "axis may not be zero. See ReduceSumByRotation()");
      }

      // Recall first dimension of a shell variant tensor is the packing
      // dimension.
      int clamped_axis = axis;
      if (clamped_axis < 0) {
        clamped_axis += rank + 1;
      } else if (clamped_axis > 0) {
        clamped_axis -= 1;
      }

      if (clamped_axis < 0 || clamped_axis > rank) {
        return InvalidArgument("axis must be in the range [0, rank], got ",
                               clamped_axis);
      }

      ShapeHandle output;

      // If this op ever supports keepdim=True, use the following shape.
      //  DimensionHandle reduced_dim = c->MakeDim({1});
      //  TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), clamped_axis,
      //  reduced_dim, &output));

      // This op currently only supports keepdim=False whose shape is computed
      // via the following.
      ShapeHandle prefix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, clamped_axis, &prefix));

      ShapeHandle postfix;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input(0), clamped_axis + 1, rank, &postfix));

      if (clamped_axis == 0) {
        output = postfix;
      } else if (clamped_axis == rank - 1) {
        output = prefix;
      } else {
        TF_RETURN_IF_ERROR(c->Concatenate(prefix, postfix, &output));
      }

      c->set_output(0, output);
      return OkStatus();
    });

// Modulus switching.
REGISTER_OP("ModulusReduceContext64")
    .Input("context: variant")
    .Output("reduced_context: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("ModulusReduceKey64")
    .Input("key: variant")
    .Output("reduced_key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("ModulusReduceCt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("reduced_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("ModulusReducePt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("reduced_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

// Shape kernels.
REGISTER_OP("ExpandDimsVariant")
    .Input("value: variant")
    .Attr("axis: int")
    .Output("expanded_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 rank = c->Rank(c->input(0));

      tsl::int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

      // Check that axis is in the correct range.
      if (axis < -rank || axis > rank) {
        return InvalidArgument("axis must be in the range [-rank, rank], got ",
                               axis);
      }

      if (axis < 0) {
        axis += rank;
      }

      ShapeHandle prefix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, axis, &prefix));

      ShapeHandle postfix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), axis, rank - 1, &postfix));

      ShapeHandle output;
      ShapeHandle axis_dim = c->MakeShape({1});
      TF_RETURN_IF_ERROR(c->Concatenate(prefix, axis_dim, &output));
      TF_RETURN_IF_ERROR(c->Concatenate(output, postfix, &output));

      c->set_output(0, output);
      return OkStatus();
    });
