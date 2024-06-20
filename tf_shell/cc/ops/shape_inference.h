#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::Status;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

Status ImportAndRemoveBatchingDimShape(InferenceContext* c);

template <unsigned int ArgNum>
Status ExportAndAddBatchingDimShape(InferenceContext* c) {
  tsl::int32 batching_dim;
  TF_RETURN_IF_ERROR(c->GetAttr("batching_dim", &batching_dim));
  ShapeHandle batching_dim_shape = c->MakeShape({batching_dim});

  ShapeHandle output;
  TF_RETURN_IF_ERROR(
      c->Concatenate(batching_dim_shape, c->input(ArgNum), &output));

  c->set_output(0, output);
  return OkStatus();
}

template <unsigned int ArgNum>
Status UnchangedArgShape(InferenceContext* c) {
  c->set_output(0, c->input(ArgNum));
  return OkStatus();
}

Status ShellBroadcastingOpShape(InferenceContext* c);

Status ShellMatMulCtPtShape(InferenceContext* c);

Status ShellMatMulPtCtShape(InferenceContext* c);

Status ShellSegmentReductionWithNumSegmentsShape(InferenceContext* c);