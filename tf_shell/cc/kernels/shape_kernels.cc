/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
// #include "tensorflow/core/framework/variant_op_registry.h"
// #include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/variant.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_VARIANT;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;

// This class is exactly like TensorFlow's ExpandDimsOp, but allows operating
// on a tensor with a variant dtype.
class ExpandDimsVariantOp : public OpKernel {
 public:
  explicit ExpandDimsVariantOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // OP_REQUIRES(ctx, ctx->input(0).dtype() != DT_VARIANT,
    //             InvalidArgument("ExpandDims on Variant not supported"));

    int32 dim = ctx->input(1).flat<int32>()(0);

    // Recall first dimension of a shell variant tensor is the packing
    // dimension. We don't allow expanding this dimension.
    OP_REQUIRES(ctx, dim != 0, InvalidArgument("Invalid dimension index."));
    dim += dim > 0 ? -1 : 0;

    OP_REQUIRES(
        ctx, (dim >= -1 - ctx->input(0).dims() && dim <= ctx->input(0).dims()),
        InvalidArgument("Tried to expand dim index ", dim, " for tensor with ",
                        ctx->input(0).dims(), " dimensions."));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    // Safe - # elements in tensor dims bounded.
    int const existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64> new_shape(existing_dims_size);
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      dim += existing_dims.size() + 1;
    }

    // Clamp to the end if needed.
    dim = std::min<int32>(dim, existing_dims_size);
    new_shape.emplace(new_shape.begin() + dim, 1);
    TensorShape const output_shape(new_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and output
      // should always be the same (we only expand the dimension with 1).
      ctx->SetStatus(Internal("Could not expand dimension with input shape ",
                              ctx->input(0).shape().DebugString(),
                              " and output shape ",
                              output_shape.DebugString()));
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("ExpandDimsVariant").Device(DEVICE_CPU),
                        ExpandDimsVariantOp);
