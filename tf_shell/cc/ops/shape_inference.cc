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
// #include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::BroadcastBinaryOpOutputShapeFnHelper;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ScalarShape;
using tensorflow::shape_inference::ShapeHandle;

Status ImportAndRemoveBatchingDimShape(InferenceContext* c) {
  ShapeHandle output;

  // First dimension of "in" is stored via shell Polynomial.
  TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 1, &output));

  c->set_output(0, output);
  return OkStatus();
}

Status ShellBroadcastingOpShape(InferenceContext* c) {
  if (c->num_inputs() != 3) {
    return InvalidArgument("Expected 3 inputs but got: ", c->num_inputs());
  }

  ShapeHandle a_shape = c->input(1);
  ShapeHandle b_shape = c->input(2);
  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(c, a_shape, b_shape,
                                                          true, &output_shape));
  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellMatMulCtPtShape(InferenceContext* c) {
  ShapeHandle a_shape;  // a is the ciphertext with batch axis packing.
  ShapeHandle b_shape;  // b is the plaintext.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &b_shape));

  // When the ciphertext a is rank 1, it is still considered a matrix
  // because the first axis is the packing dimension.
  bool a_batched = c->Rank(a_shape) > 1;

  // Determine output rows and columns.
  DimensionHandle output_rows =
      a_batched ? c->Dim(a_shape, -2) : c->MakeDim({1});
  DimensionHandle output_cols = c->Dim(b_shape, -1);

  // Inner dimensions should be compatible.
  DimensionHandle inner_merged;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(a_shape, -1), c->Dim(b_shape, -2), &inner_merged));

  // Batch dimensions should broadcast with each other.
  ShapeHandle a_batch_shape;
  ShapeHandle b_batch_shape;
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(
      c->Subshape(a_shape, 0, a_batched ? -2 : -1, &a_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_shape));

  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, a_batch_shape, b_batch_shape, true, &output_batch_shape));

  ShapeHandle output_shape;
  if (a_batched) {
    TF_RETURN_IF_ERROR(c->Concatenate(output_batch_shape,
                                      c->Matrix(output_rows, output_cols),
                                      &output_shape));
  } else {
    // If a is not batched, the first matrix dimension is the packing dimension.
    TF_RETURN_IF_ERROR(c->Concatenate(output_batch_shape,
                                      c->Vector(output_cols), &output_shape));
  }

  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellMatMulPtCtShape(InferenceContext* c) {
  ShapeHandle a_shape;  // a is the plaintext.
  ShapeHandle b_shape;  // b is the ciphertext with batch axis packing.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &b_shape));

  // Determine output rows and columns.
  DimensionHandle output_rows = c->Dim(a_shape, -2);
  DimensionHandle output_cols = c->Dim(b_shape, -1);

  // Inner dimensions compatibility checked at runtime.
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &output_batch_shape));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(
      output_batch_shape, c->Matrix(output_rows, output_cols), &output_shape));

  c->set_output(0, output_shape);
  return OkStatus();
}

// Based on
// https://github.com/tensorflow/tensorflow/blob/c6e9fd55508e466aa7db265eb5742ac9e4c4332e/tensorflow/core/framework/common_shape_fns.cc#L2408
Status ShellSegmentReductionWithNumSegmentsShape(InferenceContext* c) {
  ShapeHandle s_data = c->input(1);
  ShapeHandle s_segment_ids = c->input(2);
  ShapeHandle s_num_segments = c->input(3);
  TF_RETURN_IF_ERROR(c->WithRank(s_num_segments, 0, &s_num_segments));

  ShapeHandle out;

  if (c->RankKnown(s_segment_ids)) {
    // Leading dimensions of data must be compatible with dimensions of
    // s_segment_ids, but ignore the batch axis packing dimension which
    // is checked at op kernel time.
    ShapeHandle s_segment_ids_suffix;
    TF_RETURN_IF_ERROR(c->Subshape(s_segment_ids, 1, &s_segment_ids_suffix));
    ShapeHandle matching_prefix;
    TF_RETURN_IF_ERROR(c->MergePrefix(s_data, s_segment_ids_suffix, &s_data,
                                      &s_segment_ids_suffix));

    // Get the value of the num_segments input tensor.
    DimensionHandle num_segments_dim;
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(3, &num_segments_dim));

    // Output is {2} + {ct dim} + {segment_id_rank} + s_data[segment_id_rank -
    // 1:]. 2 is because the top and bottom of the ciphertexts are treated
    // independently.
    ShapeHandle s_data_suffix;
    auto rank = c->Rank(s_segment_ids_suffix);
    TF_RETURN_IF_ERROR(c->Subshape(s_data, rank, &s_data_suffix));
    TF_RETURN_IF_ERROR(
        c->Concatenate(c->Vector(num_segments_dim), s_data_suffix, &out));
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->MakeDim(2)), out, &out));
  } else {
    out = c->UnknownShape();
  }
  c->set_output(0, out);
  return OkStatus();
}