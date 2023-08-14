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

#include <memory>

#include "prng_variant.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
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
using tensorflow::tstring;
using tensorflow::Variant;

class PrngImportOp : public OpKernel {
 private:
  using Prng = rlwe::SecurePrng;
  using HkdfPrng = rlwe::SingleThreadHkdfPrng;

 public:
  explicit PrngImportOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    tstring t_in_key;
    OP_REQUIRES_OK(op_ctx, GetScalar<tstring>(op_ctx, 0, &t_in_key));
    std::string in_key(t_in_key.c_str());

    if (in_key.empty()) {
      OP_REQUIRES_VALUE(in_key, op_ctx, HkdfPrng::GenerateSeed());
    }

    std::unique_ptr<Prng> rng;
    OP_REQUIRES_VALUE(rng, op_ctx, HkdfPrng::Create(in_key));

    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));

    PrngVariant prng_variant(std::move(rng));
    out->scalar<Variant>()() = std::move(prng_variant);
  }
};

REGISTER_KERNEL_BUILDER(Name("PrngImport").Device(DEVICE_CPU), PrngImportOp);
