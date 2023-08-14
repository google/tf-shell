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
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/polynomial.h"
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
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

template <typename From, typename To>
class PolynomialImportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<To>;
  using ModularIntParams = typename ModularInt::Params;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using Context = rlwe::RlweContext<ModularInt>;
  using NttParams = rlwe::NttParameters<ModularInt>;

 public:
  explicit PolynomialImportOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<To> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<To>>(op_ctx, 0, &shell_ctx_var));
    ModularIntParams const* pt_params = shell_ctx_var->pt_params.get();
    NttParams const* pt_ntt_params = &(shell_ctx_var->pt_ntt_params);
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    size_t num_slots = shell_ctx->GetN();

    Tensor const& input = op_ctx->input(1);

    // First dimension of the shape must equal number of slots.
    OP_REQUIRES(
        op_ctx,
        input.dims() > 0 && static_cast<size_t>(input.dim_size(0)) == num_slots,
        InvalidArgument("Dimensions expected to start with:", num_slots,
                        "but got shape:", input.shape().DebugString()));

    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.RemoveDimWithStatus(0));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_input = input.flat_outer_dims<From>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_input.dimension(1); ++i) {
      // Create a vector of montgomery ints.
      std::vector<ModularInt> mont(num_slots,
                                   ModularInt::ImportZero(pt_params));
      for (size_t slot = 0; slot < num_slots; ++slot) {
        OP_REQUIRES_VALUE(
            mont[slot], op_ctx,
            ModularInt::ImportInt(flat_input(slot, i), pt_params));
      }

      // Convert montgomery coefficients to polynomial in inverse ntt form
      // mod t, then switch modulus to q so that subsequent polynomial
      // multiplication under encryption performs point-wise multiplication in
      // plaintext. Modulus switching to the larger modulus is non-trivial when
      // plaintext numbers are negative.
      Polynomial mont_mod_t(std::move(mont));
      std::vector<ModularInt> intt_mont_mod_t =
          mont_mod_t.InverseNtt(pt_ntt_params, pt_params);

      OP_REQUIRES_VALUE(
          std::vector<ModularInt> intt_mont_mod_q, op_ctx,
          rlwe::ConvertModulusBalanced(intt_mont_mod_t, *pt_params,
                                       *shell_ctx->GetModulusParams()));

      Polynomial mont_mod_q =
          Polynomial::ConvertToNtt(intt_mont_mod_q, shell_ctx->GetNttParams(),
                                   shell_ctx->GetModulusParams());

      // Wrap in a PolynomialVariant and store Polynomial in output tensor.
      auto wrapped = PolynomialVariant<To>(std::move(mont_mod_q));
      flat_output(i) = std::move(wrapped);
    }
  }
};

template <typename From, typename To>
class PolynomialExportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<From>;
  using ModularIntParams = typename ModularInt::Params;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using Context = rlwe::RlweContext<ModularInt>;
  using NttParams = rlwe::NttParameters<ModularInt>;

 public:
  explicit PolynomialExportOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<From> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<From>>(op_ctx, 0, &shell_ctx_var));
    ModularIntParams const* pt_params = shell_ctx_var->pt_params.get();
    NttParams const* pt_ntt_params = &(shell_ctx_var->pt_ntt_params);
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    size_t num_slots = shell_ctx->GetN();

    Tensor const& input = op_ctx->input(1);

    // set output dim to include slots in the polynomial
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.InsertDimWithStatus(0, num_slots));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_input = input.flat<Variant>();
    auto flat_output = output->flat_outer_dims<To>();

    for (int i = 0; i < flat_output.dimension(1); ++i) {
      PolynomialVariant<From> const* pv =
          std::move(flat_input(i).get<PolynomialVariant<From>>());
      OP_REQUIRES(op_ctx, pv != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  "did not unwrap successfully."));
      Polynomial p_mod_q = std::move(pv->poly);

      OP_REQUIRES(
          op_ctx, p_mod_q.Coeffs().size() == num_slots,
          InvalidArgument("Polynomial dimensions:", p_mod_q.Coeffs().size(),
                          "do not match shell context degree:", num_slots));

      // Shell Polynomial is in ntt form, mod q. First inverse ntt (mod q), then
      // switch from mod q back to mod t, then compute the ntt to recover the
      // original input.
      std::vector<ModularInt> intt_p_mod_q = p_mod_q.InverseNtt(
          shell_ctx->GetNttParams(), shell_ctx->GetModulusParams());

      OP_REQUIRES_VALUE(
          std::vector<ModularInt> intt_p_mod_t, op_ctx,
          rlwe::ConvertModulusBalanced(
              intt_p_mod_q, *shell_ctx->GetModulusParams(), *pt_params));

      Polynomial p_mod_t =
          Polynomial::ConvertToNtt(intt_p_mod_t, pt_ntt_params, pt_params);

      // convert out of montgomery form and store in output tensor
      std::vector<ModularInt> mont = p_mod_t.Coeffs();
      for (size_t slot = 0; slot < num_slots; ++slot) {
        To res = mont[slot].ExportInt(pt_params);
        res = fix_sign_extend(res, shell_ctx->GetLogT());
        flat_output(slot, i) = res;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint8>("dtype"),
                        PolynomialImportOp<uint8, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("PolynomialImport64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    PolynomialImportOp<int8, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int16>("dtype"),
                        PolynomialImportOp<int16, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("dtype"),
                        PolynomialImportOp<int32, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("dtype"),
                        PolynomialImportOp<int64, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("dtype"),
                        PolynomialImportOp<float, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("dtype"),
                        PolynomialImportOp<double, uint64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint8>("dtype"),
                        PolynomialExportOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(
    Name("PolynomialExport64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    PolynomialExportOp<uint64, int8>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int16>("dtype"),
                        PolynomialExportOp<uint64, int16>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("dtype"),
                        PolynomialExportOp<uint64, int32>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("dtype"),
                        PolynomialExportOp<uint64, int64>);