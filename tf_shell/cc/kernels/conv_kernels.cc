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
#include "polynomial_variant.h"
#include "rotation_variants.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;

template <typename T>
static inline StatusOr<rlwe::RnsBgvCiphertext<rlwe::MontgomeryInt<T>>> OpCore(
    SymmetricCtVariant<T> const* a, PolynomialVariant<T> const* b,
    ContextVariant<T> const* shell_ctx_var) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

  if (TF_PREDICT_FALSE(a == nullptr)) {
    return InvalidArgument("Ciphertext input a is null.");
  }
  TF_RETURN_IF_ERROR(const_cast<SymmetricCtVariant<T>*>(a)->MaybeLazyDecode(
      shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
  SymmetricCt const& a_ct = a->ct;

  if (TF_PREDICT_FALSE(b == nullptr)) {
    return InvalidArgument("Polynomial input b is null.");
  }
  TF_RETURN_IF_ERROR(const_cast<PolynomialVariant<T>*>(b)->MaybeLazyDecode(
      shell_ctx_var->ct_context_));
  RnsPolynomial const& b_poly = b->poly;

  SymmetricCt res = a_ct;
  Status s = res.AbsorbInPlace(b_poly);
  if (TF_PREDICT_FALSE(!s.ok())) {
    return s;
  }
  return res;
}

template <typename T>
static inline StatusOr<rlwe::RnsBgvCiphertext<rlwe::MontgomeryInt<T>>> OpCore(
    PolynomialVariant<T> const* a, SymmetricCtVariant<T> const* b,
    ContextVariant<T> const* shell_ctx_var) {
  return OpCore(b, a, shell_ctx_var);
}

template <typename T>
static inline StatusOr<rlwe::RnsBgvCiphertext<rlwe::MontgomeryInt<T>>> OpCore(
    SymmetricCtVariant<T> const* a, SymmetricCtVariant<T> const* b,
    ContextVariant<T> const* shell_ctx_var) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

  if (TF_PREDICT_FALSE(a == nullptr)) {
    return InvalidArgument("Ciphertext input a is null.");
  }
  TF_RETURN_IF_ERROR(const_cast<SymmetricCtVariant<T>*>(a)->MaybeLazyDecode(
      shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
  SymmetricCt const& a_ct = a->ct;

  if (TF_PREDICT_FALSE(b == nullptr)) {
    return InvalidArgument("Ciphertext input b is null.");
  }
  TF_RETURN_IF_ERROR(const_cast<SymmetricCtVariant<T>*>(b)->MaybeLazyDecode(
      shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
  SymmetricCt const& b_ct = b->ct;

  return a_ct * b_ct;
}

// This Op can multiply either a shell ciphertext or a plaintext polynomial by
// a plaintext scalar, depending on the class template.
template <typename T, typename InputCtOrPoly, typename FilterCtOrPoly,
          bool AllowDifferentNumInChannels>
class Conv2dOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using ErrorParams = rlwe::RnsErrorParams<ModularInt>;

  std::vector<tsl::int32> stride;
  std::vector<tsl::int32> padding;
  std::vector<tsl::int32> dilation;

 public:
  explicit Conv2dOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    // Get the strides and padding attributes.
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("strides", &stride));
    OP_REQUIRES(op_ctx, stride.size() == 4,
                InvalidArgument("Strides must have 4 elements."));

    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("padding", &padding));
    OP_REQUIRES(op_ctx, padding.size() == 4,
                InvalidArgument("Padding must have 4 elements."));

    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("dilations", &dilation));
    OP_REQUIRES(op_ctx, dilation.size() == 4,
                InvalidArgument("dilations must have 4 elements."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Polynomial polynomial convolution is not implemented.
    if constexpr (std::is_same<InputCtOrPoly, PolynomialVariant<T>>::value &&
                  std::is_same<FilterCtOrPoly, PolynomialVariant<T>>::value) {
      OP_REQUIRES(op_ctx, false,
                  InvalidArgument(
                      "Polynomial * polynomial convolution not implemented."));
    }

    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    uint64_t const num_slots = ((uint64_t)1) << shell_ctx_var->log_n_;

    Tensor const& x = op_ctx->input(1);
    Tensor const& filter = op_ctx->input(2);

    // x has shape [batch size (implicit), height, width, in_channels]
    OP_REQUIRES(op_ctx, x.dims() == 3,
                InvalidArgument("Input x must have 3 dimensions."));
    int64_t const height = x.dim_size(0);
    int64_t const width = x.dim_size(1);
    int64_t const in_channels = x.dim_size(2);
    auto shaped_x = x.shaped<Variant, 3>({height, width, in_channels});

    // filter has shape [height, width, in_channels, out_channels]
    OP_REQUIRES(op_ctx, filter.dims() == 4,
                InvalidArgument("Filter must have 4 dimensions."));
    int64_t const filter_height = filter.dim_size(0);
    int64_t const filter_width = filter.dim_size(1);
    int64_t const filter_in_channels = filter.dim_size(2);
    int64_t const filter_out_channels = filter.dim_size(3);
    auto shaped_filter = filter.shaped<Variant, 4>(
        {filter_height, filter_width, filter_in_channels, filter_out_channels});

    if constexpr (!AllowDifferentNumInChannels) {
      OP_REQUIRES(
          op_ctx, filter_in_channels == in_channels,
          InvalidArgument(
              "Input x and filter must have the same number of channels."));
    }

    // Stride is a tensor of shape [4].
    int64_t const stride_batch = stride[0];
    int64_t const stride_height = stride[1];
    int64_t const stride_width = stride[2];
    int64_t const stride_in_channels = stride[3];
    OP_REQUIRES(op_ctx, stride_batch == 1 && stride_in_channels == 1,
                InvalidArgument("Batch and in_channels strides must be 1."));

    // Padding is a tensor of shape [4].
    int64_t const padding_top = padding[0];
    int64_t const padding_bottom = padding[1];
    int64_t const padding_left = padding[2];
    int64_t const padding_right = padding[3];
    OP_REQUIRES(op_ctx,
                padding_top >= 0 && padding_bottom >= 0 && padding_left >= 0 &&
                    padding_right >= 0,
                InvalidArgument("Padding must be non-negative."));
    OP_REQUIRES(op_ctx,
                padding_top < filter_height && padding_bottom < filter_height,
                InvalidArgument("Padding must be less than filter height."));
    OP_REQUIRES(op_ctx,
                padding_left < filter_width && padding_right < filter_width,
                InvalidArgument("Padding must be less than filter width."));

    int64_t const dilation_batch = dilation[0];
    int64_t const dilation_height = dilation[1];
    int64_t const dilation_width = dilation[2];
    int64_t const dilation_in_channels = dilation[3];
    OP_REQUIRES(op_ctx, dilation_batch == 1 && dilation_in_channels == 1,
                InvalidArgument("Batch and in_channels dilations must be 1."));
    OP_REQUIRES(
        op_ctx, dilation_height > 0 && dilation_width > 0,
        InvalidArgument("Height and width dilations must be positive."));
    OP_REQUIRES(op_ctx, dilation_height < height,
                InvalidArgument("Height dilation must be less than height."));
    OP_REQUIRES(op_ctx, dilation_width < width,
                InvalidArgument("Width dilation must be less than width."));
    int64_t const filter_dilated_height =
        (filter_height - 1) * dilation_height + 1;
    int64_t const filter_dilated_width =
        (filter_width - 1) * dilation_width + 1;

    int64_t const h_start = -padding_top;
    int64_t const h_end = height + padding_bottom - filter_dilated_height;
    int64_t const w_start = -padding_left;
    int64_t const w_end = width + padding_right - filter_dilated_width;
    int64_t const c_start = 0;
    int64_t const c_end = in_channels - filter_in_channels;

    // Allocate output with shape
    //   [batch size (implicit), out_height, out_width, out_channels]
    int64_t const out_height = (h_end - h_start) / stride_height + 1;
    int64_t const out_width = (w_end - w_start) / stride_width + 1;
    int64_t const out_channels = (c_end - c_start) / stride_in_channels + 1;
    Tensor* output;
    TensorShape output_shape;
    if constexpr (AllowDifferentNumInChannels) {
      output_shape = {out_height, out_width, out_channels, filter_out_channels};
    } else {
      output_shape = {out_height, out_width, filter_out_channels};
    }
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto shaped_output = output->shaped<Variant, 4>(
        {out_height, out_width, out_channels, filter_out_channels});

    // Perform the convolution by sliding over the input tensor. Parallelize
    // over the height dimension.
    auto convolve_in_range = [&](int64_t start, int64_t end) {
      // Introduce the stride and offset into the work indices.
      int thread_h_start = (start * stride_height) + h_start;
      int thread_h_end = (end * stride_height) + h_start;

      for (int64_t h = thread_h_start; h < thread_h_end; h += stride_height) {
        for (int64_t w = w_start; w <= w_end; w += stride_width) {
          for (int64_t c = c_start; c <= c_end; c += stride_in_channels) {
            // Matmul a 1D patch of the input with the 2d filter of shape
            // [height * width * in_channels, out_channels]. The 1D input patch
            // is flattened to shape [filter_height * filter_width *
            // in_channels]. The flattened patches are not materialized and the
            // matrix multiplication is done with indexing.
            for (int64_t o = 0; o < filter_out_channels; ++o) {
              // Compute the dot product of the 1D input patch
              // [height * width * in_channels] by the 2d filter flattened in a
              // similar way [height * width * in_channels, out_channels].
              //
              // `dot_product` will hold the running sum, initialize with the
              // first result of i=0, j=0, k=0.
              SymmetricCt* dot_product = nullptr;
              InputCtOrPoly const* x_val = nullptr;
              FilterCtOrPoly const* filter_val = nullptr;

              for (int64_t i = 0; i < filter_height; ++i) {
                for (int64_t j = 0; j < filter_width; ++j) {
                  for (int64_t k = 0; k < filter_in_channels; ++k) {
                    int64_t const in_i = h + (i * dilation_height);
                    int64_t const in_j = w + (j * dilation_width);
                    // Same effect as zero padding the inner edge.
                    if (in_i < 0 || in_j < 0) {
                      continue;
                    }
                    // Same effect as zero padding the outer edge.
                    if (in_i >= height || in_j >= width) {
                      continue;
                    }

                    // Get the inputs.
                    x_val = shaped_x(in_i, in_j, c + k).get<InputCtOrPoly>();
                    filter_val =
                        shaped_filter(i, j, k, o).get<FilterCtOrPoly>();

                    // Multiply
                    OP_REQUIRES_VALUE(SymmetricCt mul, op_ctx,
                                      OpCore(x_val, filter_val, shell_ctx_var));

                    // Add to the dot product.
                    if (dot_product == nullptr) {
                      dot_product = new SymmetricCt(std::move(mul));
                    } else {
                      OP_REQUIRES_OK(op_ctx, dot_product->AddInPlace(mul));
                    }
                  }
                }
              }  // End matrix multiplication
              OP_REQUIRES(op_ctx, dot_product != nullptr,
                          Internal("Internal error, dot product is NULL."));
              OP_REQUIRES(op_ctx, x_val != nullptr,
                          Internal("Internal error, x_val is NULL."));
              OP_REQUIRES(op_ctx, filter_val != nullptr,
                          Internal("Internal error, filter_val is NULL."));

              // Recover the ct_context and error params from one of the
              // ciphertext inputs. At least one of them is guaranteed to be a
              // ciphertext. Prefer x's in the case that both are ciphertexts
              // since that is preserved in the output in that case.
              std::shared_ptr<Context const> ct_context;
              std::shared_ptr<ErrorParams const> error_params;
              if constexpr (std::is_same<InputCtOrPoly,
                                         SymmetricCtVariant<T>>::value) {
                ct_context = x_val->ct_context;
                error_params = x_val->error_params;
              } else {
                ct_context = filter_val->ct_context;
                error_params = filter_val->error_params;
              }

              SymmetricCtVariant<T> result_var(std::move(*dot_product),
                                               std::move(ct_context),
                                               std::move(error_params));
              shaped_output((h - h_start) / stride_height,
                            (w - w_start) / stride_width,
                            (c - c_start) / stride_in_channels, o) =
                  std::move(result_var);
            }
          }
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost = 20 * out_width * out_channels * filter_height *
                     filter_width * filter_in_channels *
                     num_slots;  // ns measured on log_n = 11
    thread_pool->ParallelFor(out_height, cost, convolve_in_range);
  }
};

// This Op can multiply either a shell ciphertext or a plaintext polynomial by
// a plaintext scalar, depending on the class template.
template <typename T, typename InputCtOrPoly, typename FilterCtOrPoly,
          bool AllowDifferentNumInChannels>
class Conv2dTransposeOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using ErrorParams = rlwe::RnsErrorParams<ModularInt>;

  std::vector<tsl::int32> stride;
  std::vector<tsl::int32> padding;
  std::vector<tsl::int32> dilation;

 public:
  explicit Conv2dTransposeOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    // Get the strides and padding attributes.
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("strides", &stride));
    OP_REQUIRES(op_ctx, stride.size() == 4,
                InvalidArgument("Strides must have 4 elements."));

    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("padding", &padding));
    OP_REQUIRES(op_ctx, padding.size() == 4,
                InvalidArgument("Padding must have 4 elements."));

    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("dilations", &dilation));
    OP_REQUIRES(op_ctx, dilation.size() == 4,
                InvalidArgument("dilations must have 4 elements."));

    int64_t const dilation_batch = dilation[0];
    int64_t const dilation_height = dilation[1];
    int64_t const dilation_width = dilation[2];
    int64_t const dilation_in_channels = dilation[3];
    OP_REQUIRES(op_ctx,
                dilation_batch == 1 && dilation_height == 1 &&
                    dilation_width == 1 && dilation_in_channels == 1,
                InvalidArgument("All dilations must be 1."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Polynomial polynomial convolution is not implemented.
    if constexpr (std::is_same<InputCtOrPoly, PolynomialVariant<T>>::value &&
                  std::is_same<FilterCtOrPoly, PolynomialVariant<T>>::value) {
      OP_REQUIRES(op_ctx, false,
                  InvalidArgument(
                      "Polynomial * polynomial convolution not implemented."));
    }
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    uint64_t const num_slots = ((uint64_t)1) << shell_ctx_var->log_n_;

    Tensor const& x = op_ctx->input(1);
    Tensor const& filter = op_ctx->input(2);

    // x has shape [batch size (implicit), height, width, in_channels]
    OP_REQUIRES(op_ctx, x.dims() == 3,
                InvalidArgument("Input x must have 3 dimensions."));
    int64_t const height = x.dim_size(0);
    int64_t const width = x.dim_size(1);
    int64_t const in_channels = x.dim_size(2);
    auto shaped_x = x.shaped<Variant, 3>({height, width, in_channels});

    // filter has shape [height, width, in_channels, out_channels]
    OP_REQUIRES(op_ctx, filter.dims() == 4,
                InvalidArgument("Filter must have 4 dimensions."));
    int64_t const filter_height = filter.dim_size(0);
    int64_t const filter_width = filter.dim_size(1);
    int64_t const filter_out_channels = filter.dim_size(2);
    int64_t const filter_in_channels = filter.dim_size(3);
    auto shaped_filter = filter.shaped<Variant, 4>(
        {filter_height, filter_width, filter_out_channels, filter_in_channels});

    if constexpr (!AllowDifferentNumInChannels) {
      OP_REQUIRES(
          op_ctx, filter_in_channels == in_channels,
          InvalidArgument(
              "Input x and filter must have the same number of channels."));
    }

    // Stride is a tensor of shape [4].
    int64_t const stride_batch = stride[0];
    int64_t const stride_height = stride[1];
    int64_t const stride_width = stride[2];
    int64_t const stride_in_channels = stride[3];
    OP_REQUIRES(op_ctx, stride_batch == 1 && stride_in_channels == 1,
                InvalidArgument("Batch and in_channels strides must be 1."));
    OP_REQUIRES(
        op_ctx, stride_height < filter_height,
        InvalidArgument("Stride height must be less than filter height."));
    OP_REQUIRES(
        op_ctx, stride_width < filter_width,
        InvalidArgument("Stride width must be less than filter width."));

    // Padding is a tensor of shape [4].
    int64_t const padding_top = padding[0];
    int64_t const padding_bottom = padding[1];
    int64_t const padding_left = padding[2];
    int64_t const padding_right = padding[3];
    OP_REQUIRES(op_ctx,
                padding_top >= 0 && padding_bottom >= 0 && padding_left >= 0 &&
                    padding_right >= 0,
                InvalidArgument("Padding must be non-negative."));
    OP_REQUIRES(op_ctx,
                padding_top < filter_height && padding_bottom < filter_height,
                InvalidArgument("Padding must be less than filter height."));
    OP_REQUIRES(op_ctx,
                padding_left < filter_width && padding_right < filter_width,
                InvalidArgument("Padding must be less than filter width."));

    int64_t const dilation_batch = dilation[0];
    int64_t const dilation_height = dilation[1];
    int64_t const dilation_width = dilation[2];
    int64_t const dilation_in_channels = dilation[3];
    OP_REQUIRES(op_ctx,
                dilation_batch == 1 && dilation_height == 1 &&
                    dilation_width == 1 && dilation_in_channels == 1,
                InvalidArgument("Dilation is not yet supported."));

    int64_t const h_start = -filter_height + 1 + padding_top;
    int64_t const h_end = ((height - 1) * stride_height) - padding_bottom;
    int64_t const w_start = -filter_width + 1 + padding_left;
    int64_t const w_end = ((width - 1) * stride_width) - padding_right;
    int64_t c_start = -filter_in_channels + 1;
    int64_t c_end = ((in_channels - 1) * stride_in_channels);
    if constexpr (!AllowDifferentNumInChannels) {
      c_start = 0;
      c_end = 0;
    }

    // Allocate output with shape
    //   [batch size (implicit), out_height, out_width, out_channels]
    int64_t const out_height = h_end - h_start + 1;
    int64_t const out_width = w_end - w_start + 1;
    int64_t const out_channels = c_end - c_start + 1;
    Tensor* output;
    TensorShape output_shape;
    if constexpr (AllowDifferentNumInChannels) {
      output_shape = {out_height, out_width, out_channels, filter_out_channels};
    } else {
      output_shape = {out_height, out_width, filter_out_channels};
    }
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto shaped_output = output->shaped<Variant, 4>(
        {out_height, out_width, out_channels, filter_out_channels});

    // Perform the convolution by sliding over the input tensor.
    auto convolve_in_range = [&](int64_t start, int64_t end) {
      // Introduce the stride and offset into the work indices.
      int thread_h_start = start + h_start;
      int thread_h_end = end + h_start;

      for (int64_t h = thread_h_start; h < thread_h_end; h += 1) {
        for (int64_t w = w_start; w <= w_end; w += 1) {
          for (int64_t c = c_start; c <= c_end; c += 1) {
            // Matmul a 1D patch of the input with the 2d filter of shape
            // [height * width * in_channels, out_channels]. The 1D input patch
            // is flattened to shape [filter_height * filter_width *
            // in_channels]. The flattened patches are not materialized and the
            // matrix multiplication is done with indexing.
            for (int64_t o = 0; o < filter_out_channels; ++o) {
              // Compute the dot product of the 1D input patch
              // [height * width * in_channels] by the 2d filter flattened in a
              // similar way [height * width * in_channels, out_channels].
              //
              // `dot_product` will hold the running sum, initialize with the
              // first result of i=0, j=0, k=0.
              SymmetricCt* dot_product = nullptr;
              InputCtOrPoly const* x_val = nullptr;
              FilterCtOrPoly const* filter_val = nullptr;

              for (int64_t i = 0; i < filter_height; ++i) {
                for (int64_t j = 0; j < filter_width; ++j) {
                  for (int64_t k = 0; k < filter_in_channels; ++k) {
                    // Same as inner stride padding.
                    if ((h + i) % stride_height != 0 ||
                        (w + j) % stride_width != 0 ||
                        (c + k) % stride_in_channels != 0) {
                      continue;
                    }
                    int64_t const in_i = (h + i) / stride_height;
                    int64_t const in_j = (w + j) / stride_width;
                    int64_t const in_k = (c + k) / stride_in_channels;
                    // Same effect as zero padding the inner edge.
                    if (in_i < 0 || in_j < 0) {
                      continue;
                    }
                    // Same effect as zero padding the outer edge.
                    if (in_i >= height || in_j >= width) {
                      continue;
                    }

                    // Get the inputs.
                    x_val = shaped_x(in_i, in_j, in_k).get<InputCtOrPoly>();
                    // Transpose the filter.
                    int64_t const f_i = filter_height - i - 1;
                    int64_t const f_j = filter_width - j - 1;
                    int64_t f_k;
                    if constexpr (AllowDifferentNumInChannels) {
                      f_k = filter_in_channels - k - 1;
                    } else {
                      f_k = k;
                    }
                    filter_val =
                        shaped_filter(f_i, f_j, o, f_k).get<FilterCtOrPoly>();

                    // Multiply
                    OP_REQUIRES_VALUE(SymmetricCt mul, op_ctx,
                                      OpCore(x_val, filter_val, shell_ctx_var));

                    // Add to the dot product.
                    if (dot_product == nullptr) {
                      dot_product = new SymmetricCt(std::move(mul));
                    } else {
                      OP_REQUIRES_OK(op_ctx, dot_product->AddInPlace(mul));
                    }
                  }
                }
              }  // End matrix multiplication
              OP_REQUIRES(op_ctx, dot_product != nullptr,
                          Internal("Internal error, dot product is NULL."));
              OP_REQUIRES(op_ctx, x_val != nullptr,
                          Internal("Internal error, x_val is NULL."));
              OP_REQUIRES(op_ctx, filter_val != nullptr,
                          Internal("Internal error, filter_val is NULL."));

              // Recover the ct_context and error params from one of the
              // ciphertext inputs. At least one of them is guaranteed to be a
              // ciphertext. Prefer x's in the case that both are ciphertexts
              // since that is preserved in the output in that case.
              std::shared_ptr<Context const> ct_context;
              std::shared_ptr<ErrorParams const> error_params;
              if constexpr (std::is_same<InputCtOrPoly,
                                         SymmetricCtVariant<T>>::value) {
                ct_context = x_val->ct_context;
                error_params = x_val->error_params;
              } else {
                ct_context = filter_val->ct_context;
                error_params = filter_val->error_params;
              }

              SymmetricCtVariant<T> result_var(std::move(*dot_product),
                                               std::move(ct_context),
                                               std::move(error_params));
              shaped_output(h - h_start, w - w_start, c - c_start, o) =
                  std::move(result_var);
            }
          }
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost = 20 * out_width * out_channels * filter_height *
                     filter_width * filter_in_channels *
                     num_slots;  // ns measured on log_n = 11
    thread_pool->ParallelFor(out_height, cost, convolve_in_range);
  }
};

REGISTER_KERNEL_BUILDER(Name("Conv2dPtCt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, PolynomialVariant<uint64>,
                                 SymmetricCtVariant<uint64_t>, false>);

REGISTER_KERNEL_BUILDER(Name("Conv2dCtPt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, SymmetricCtVariant<uint64_t>,
                                 PolynomialVariant<uint64>, false>);

REGISTER_KERNEL_BUILDER(Name("Conv2dCtCt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, SymmetricCtVariant<uint64_t>,
                                 SymmetricCtVariant<uint64_t>, false>);

REGISTER_KERNEL_BUILDER(Name("Conv2dWithChanPtCt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, PolynomialVariant<uint64>,
                                 SymmetricCtVariant<uint64_t>, true>);

REGISTER_KERNEL_BUILDER(Name("Conv2dWithChanCtPt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, SymmetricCtVariant<uint64_t>,
                                 PolynomialVariant<uint64>, true>);

REGISTER_KERNEL_BUILDER(Name("Conv2dWithChanCtCt64").Device(DEVICE_CPU),
                        Conv2dOp<uint64, SymmetricCtVariant<uint64_t>,
                                 SymmetricCtVariant<uint64_t>, true>);

REGISTER_KERNEL_BUILDER(Name("Conv2dTransposePtCt64").Device(DEVICE_CPU),
                        Conv2dTransposeOp<uint64, PolynomialVariant<uint64>,
                                          SymmetricCtVariant<uint64_t>, false>);

REGISTER_KERNEL_BUILDER(Name("Conv2dTransposeCtPt64").Device(DEVICE_CPU),
                        Conv2dTransposeOp<uint64, SymmetricCtVariant<uint64_t>,
                                          PolynomialVariant<uint64>, false>);

REGISTER_KERNEL_BUILDER(Name("Conv2dTransposeCtCt64").Device(DEVICE_CPU),
                        Conv2dTransposeOp<uint64, SymmetricCtVariant<uint64_t>,
                                          SymmetricCtVariant<uint64_t>, false>);

REGISTER_KERNEL_BUILDER(
    Name("Conv2dTransposeWithChanPtCt64").Device(DEVICE_CPU),
    Conv2dTransposeOp<uint64, PolynomialVariant<uint64>,
                      SymmetricCtVariant<uint64_t>, true>);

REGISTER_KERNEL_BUILDER(
    Name("Conv2dTransposeWithChanCtPt64").Device(DEVICE_CPU),
    Conv2dTransposeOp<uint64, SymmetricCtVariant<uint64_t>,
                      PolynomialVariant<uint64>, true>);

REGISTER_KERNEL_BUILDER(
    Name("Conv2dTransposeWithChanCtCt64").Device(DEVICE_CPU),
    Conv2dTransposeOp<uint64, SymmetricCtVariant<uint64_t>,
                      SymmetricCtVariant<uint64_t>, true>);