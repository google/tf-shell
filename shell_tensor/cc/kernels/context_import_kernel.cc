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

#include "context_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/polynomial.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::uint64;
using tensorflow::Variant;

template <typename T>
class ContextImportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using Context = rlwe::RlweContext<ModularInt>;
  using ContextParams = struct Context::Parameters;
  using Polynomial = rlwe::Polynomial<ModularInt>;

 public:
  explicit ContextImportOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> ctx_variant{};

    ContextParams c;
    OP_REQUIRES_OK(op_ctx, GetScalar<T>(op_ctx, 1, &c.modulus));
    OP_REQUIRES_OK(op_ctx, GetScalar<size_t>(op_ctx, 2, &c.log_n));
    OP_REQUIRES_OK(op_ctx, GetScalar<size_t>(op_ctx, 3, &c.log_t));
    OP_REQUIRES_OK(op_ctx, GetScalar<size_t>(op_ctx, 4, &c.variance));
    OP_REQUIRES_VALUE(ctx_variant.ct_context, op_ctx, Context::Create(c));

    T pt_modulus = 0;
    OP_REQUIRES_OK(op_ctx, GetScalar<T>(op_ctx, 0, &pt_modulus));
    OP_REQUIRES_VALUE(ctx_variant.pt_params, op_ctx,
                      ModularIntParams::Create(pt_modulus));

    OP_REQUIRES_VALUE(
        ctx_variant.pt_ntt_params, op_ctx,
        rlwe::InitializeNttParameters<ModularInt>(
            ctx_variant.ct_context->GetLogN(), ctx_variant.pt_params.get()));

    Tensor* out0;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out0));
    out0->scalar<Variant>()() = std::move(ctx_variant);
  }
};

REGISTER_KERNEL_BUILDER(Name("ContextImport64").Device(DEVICE_CPU),
                        ContextImportOp<uint64>);
