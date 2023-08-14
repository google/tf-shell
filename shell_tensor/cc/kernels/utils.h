/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant.h"

using tensorflow::OkStatus;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShapeUtils;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

template <typename T>
Status GetScalar(OpKernelContext* ctx, int index, T* res) {
  Tensor const& input = ctx->input(index);

  if (!TensorShapeUtils::IsScalar(input.shape())) {
    return InvalidArgument("Input must be scalar tensor");
  }

  T val = input.scalar<T>()(0);

  *res = val;
  return OkStatus();
}

template <typename To>
Status GetVariant(OpKernelContext* ctx, int index, To const** res) {
  Tensor const& input = ctx->input(index);

  if (!TensorShapeUtils::IsScalar(input.shape())) {
    return InvalidArgument("Input must be scalar tensor");
  }

  To const* t = input.scalar<Variant>()().get<To>();
  if (t == nullptr) {
    return InvalidArgument(
        "Input tensor is not the correct variant type. Saw: '",
        input.scalar<Variant>()().DebugString(), "'");
  }

  *res = t;
  return OkStatus();
}

// Calling MontgomeryInt::ExportInt() will return a value in the range
// [0, 2^log_t]. It may seem like it should be between [0, 2^log_t - 1] but that
// is not the case! This is a helper function to sign extend for this unique
// range since it is not a simple cast.
template <typename To>
inline To fix_sign_extend(To x, size_t log_t) {
  // Note ULL literals used below must be the same size or larger than the
  // largest supported data width.
  static_assert(sizeof(-1ULL) >= sizeof(To));

  static const To sign_extension_mask = static_cast<To>(-1ULL << (log_t));

  static const To top_bit = static_cast<To>(1ULL << (log_t - 1));

  static const To negative_one = static_cast<To>(1ULL << (log_t));

  if (std::is_signed<To>::value && (x & top_bit)) {
    x -= 1;
    x |= sign_extension_mask;
  } else if (std::is_signed<To>::value && (x == negative_one)) {
    x = -1;
  }
  return x;
}