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

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/util/bcast.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

using tensorflow::BCast;
using tensorflow::OkStatus;
using tensorflow::OpKernelContext;
using tensorflow::StatusOr;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::TTypes;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;

using std::vector;

// Given an OpKernelContext and an index, returns the scalar value at that index
// in the context. If the tensor is not a scalar, returns an error.
template <typename T>
StatusOr<T> GetScalar(OpKernelContext* ctx, int index) {
  Tensor const& input = ctx->input(index);

  if (!TensorShapeUtils::IsScalar(input.shape())) {
    return InvalidArgument("Input must be scalar tensor");
  }

  return input.scalar<T>()(0);
}

// Given an OpKernelContext and an index, returns the vector value at that index
// in the context. If the tensor is not a vector, returns an error.
template <typename T>
StatusOr<vector<T>> GetVector(OpKernelContext* ctx, int index) {
  Tensor const& input = ctx->input(index);

  if (!TensorShapeUtils::IsVector(input.shape())) {
    return InvalidArgument("Input must be vector tensor");
  }

  size_t n = input.NumElements();
  auto in_vec = input.vec<T>();

  vector<T> res;
  res.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    res.push_back(in_vec(i));
  }

  return std::move(res);
}

// Given an OpKernelContext and an index, returns the Variant value at that
// index in the context. If the tensor is not a Variant or does not unpack
// correctly, returns an error.
template <typename T>
StatusOr<T const*> GetVariant(OpKernelContext* ctx, int index) {
  Tensor const& input = ctx->input(index);

  if (!TensorShapeUtils::IsScalar(input.shape())) {
    return InvalidArgument("Input must be scalar tensor");
  }

  T const* t = input.scalar<Variant>()().get<T>();
  if (t == nullptr) {
    return InvalidArgument(
        "Input tensor is not the correct variant type. Saw: '",
        input.scalar<Variant>()().DebugString(), "'");
  }
  return t;
}

template <int NDIMS>
inline Eigen::Tensor<Variant, 1, Eigen::RowMajor, Eigen::DenseIndex> BFlat(
    OpKernelContext* op_ctx, Tensor const& t, BCast::Vec const& x_reshape,
    BCast::Vec const& x_bcast) {
  // A TensorFlow is a TTypes<T, NDIM>::Tensor (aka Eigen::TensorMap).
  // Eigen::TensorMap is a view into an Eigen::Tensor. When performing a
  // reshape, broadcast, or even eval on an Eigen::TensorMap, it cannot be
  // assigned to another Eigen::TensorMap. This is why the following code
  // assigns the result of the reshape to an Eigen::Tensor.
  //
  // For a demo, see https://godbolt.org/z/41xvWvb63
  typedef Eigen::Tensor<Variant, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>
      ETensor;

  ETensor reshaped_t = t.template shaped<Variant, NDIMS>(x_reshape);

  ETensor broadcasted_t =
      reshaped_t.broadcast(BCast::ToIndexArray<NDIMS>(x_bcast));

  return std::move(
      broadcasted_t.reshape(BCast::ToIndexArray<1>({broadcasted_t.size()})));
}

inline Eigen::Tensor<Variant, 1, Eigen::RowMajor, Eigen::DenseIndex> MyBFlat(
    OpKernelContext* op_ctx, Tensor const& t, BCast::Vec const& x_reshape,
    BCast::Vec const& x_bcast) {
  // Uses the switch statement approach as in:
  // `tensorflow/tensorflow/core/kernels/broadcast_to_op.h`
  int const ndims = x_reshape.size();
  switch (ndims) {
    case 1:
      return std::move(BFlat<1>(op_ctx, t, x_reshape, x_bcast));
    case 2:
      return std::move(BFlat<2>(op_ctx, t, x_reshape, x_bcast));
    case 3:
      return std::move(BFlat<3>(op_ctx, t, x_reshape, x_bcast));
    case 4:
      return std::move(BFlat<4>(op_ctx, t, x_reshape, x_bcast));
    case 5:
      return std::move(BFlat<5>(op_ctx, t, x_reshape, x_bcast));
    case 6:
      return std::move(BFlat<6>(op_ctx, t, x_reshape, x_bcast));
    default:
      op_ctx->SetStatus(Unimplemented("Broadcast ", t.DebugString(),
                                      " is not supported yet."));
      return std::move(BFlat<1>(op_ctx, t, x_reshape, x_bcast));
  }
}

// Status macros from
// https://github.com/abseil/abseil-cpp/issues/976#issuecomment-1664601671
//
//
// Run a command that returns a absl::Status.  If the called code returns an
// error status, return that status up out of this method too.
//
// Example:
//   RETURN_IF_ERROR(DoThings(4));
#define TF_SHELL_RETURN_IF_ERROR(expr)                                       \
  do {                                                                       \
    /* Using _status below to avoid capture problems if expr is "status". */ \
    const absl::Status _status = (expr);                                     \
    if (_TF_SHELL_PREDICT_FALSE(!_status.ok())) return _status;              \
  } while (0)

// Run a command that returns a absl::StatusOr<T>.  If the called code returns
// an error status, return that status up out of this method too.
//
// Example:
//   ASSIGN_OR_RETURN(auto value, MaybeGetValue(arg));
#define TF_SHELL_ASSIGN_OR_RETURN(...)                                \
  TF_SHELL_STATUS_MACROS_IMPL_GET_VARIADIC_(                          \
      (__VA_ARGS__, TF_SHELL_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_)) \
  (__VA_ARGS__)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

#define TF_SHELL_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_(_1, _2, NAME, ...) NAME
#define TF_SHELL_STATUS_MACROS_IMPL_GET_VARIADIC_(args) \
  TF_SHELL_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_ args

#define TF_SHELL_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr)          \
  TF_SHELL_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(                             \
      TF_SHELL_STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs,  \
      rexpr,                                                                 \
      return std::move(TF_SHELL_STATUS_MACROS_IMPL_CONCAT_(_status_or_value, \
                                                           __LINE__))        \
          .status())

#define TF_SHELL_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr, \
                                                      error_expression)     \
  auto statusor = (rexpr);                                                  \
  if (_TF_SHELL_PREDICT_FALSE(!statusor.ok())) {                            \
    error_expression;                                                       \
  }                                                                         \
  lhs = std::move(statusor).value()

// Internal helper for concatenating macro values.
#define TF_SHELL_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define TF_SHELL_STATUS_MACROS_IMPL_CONCAT_(x, y) \
  TF_SHELL_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

// Internal helper for stringifying macro values.
#define _TF_SHELL_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
