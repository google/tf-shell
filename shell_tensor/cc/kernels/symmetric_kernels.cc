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
#include "prng_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/symmetric_encryption.h"
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
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

template <typename T>
class KeyGenOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using KeyClass = rlwe::SymmetricRlweKey<ModularInt>;

 public:
  explicit KeyGenOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<T>>(op_ctx, 0, &shell_ctx_var));
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    PrngVariant const* prng_var;
    OP_REQUIRES_OK(op_ctx, GetVariant<PrngVariant>(op_ctx, 1, &prng_var));
    Prng* prng = prng_var->prng.get();

    std::unique_ptr<const Context> shell_context;

    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));

    // KeyClass k;
    OP_REQUIRES_VALUE(
        auto k, op_ctx,
        KeyClass::Sample(shell_ctx->GetLogN(), shell_ctx->GetVariance(),
                         shell_ctx->GetLogT(), shell_ctx->GetModulusParams(),
                         shell_ctx->GetNttParams(), prng));

    SymmetricKeyVariant<T> key_variant(std::move(k));
    out->scalar<Variant>()() = std::move(key_variant);
  }
};

template <typename T>
class EncryptOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using KeyClass = rlwe::SymmetricRlweKey<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit EncryptOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<T>>(op_ctx, 0, &shell_ctx_var));
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    PrngVariant const* prng_var;
    OP_REQUIRES_OK(op_ctx, GetVariant<PrngVariant>(op_ctx, 1, &prng_var));
    Prng* prng = prng_var->prng.get();

    SymmetricKeyVariant<T> const* key_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<SymmetricKeyVariant<T>>(op_ctx, 2, &key_var));
    KeyClass const* key = &key_var->key;

    Tensor const& input = op_ctx->input(3);

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, input.shape(), &output));

    auto flat_input = input.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pv =
          std::move(flat_input(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  "did not unwrap successfully."));
      Polynomial const& p = pv->poly;

      OP_REQUIRES_VALUE(CtClass ct, op_ctx,
                        rlwe::Encrypt<ModularInt>(
                            *key, p, shell_ctx->GetErrorParams(), prng));

      SymmetricCtVariant ct_var(std::move(ct));
      flat_output(i) = std::move(ct_var);
    }
  }
};

template <typename From, typename To>
class DecryptOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<From>;
  using ModularIntParams = typename rlwe::MontgomeryInt<From>::Params;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using KeyClass = rlwe::SymmetricRlweKey<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit DecryptOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<From> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<From>>(op_ctx, 0, &shell_ctx_var));
    ModularIntParams const* pt_params = shell_ctx_var->pt_params.get();
    NttParams const* pt_ntt_params = &(shell_ctx_var->pt_ntt_params);
    Context const* shell_ctx = shell_ctx_var->ct_context.get();
    size_t num_slots = shell_ctx->GetN();

    SymmetricKeyVariant<From> const* key_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<SymmetricKeyVariant<From>>(op_ctx, 1, &key_var));
    KeyClass const* key = &key_var->key;

    Tensor const& input = op_ctx->input(2);
    OP_REQUIRES(op_ctx, input.dim_size(0) > 0,
                InvalidArgument("Cannot decrypt empty ciphertext"));
    auto flat_input = input.flat<Variant>();

    // Set output shape to include polynomial slots as first dim.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.InsertDimWithStatus(0, num_slots));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_output = output->flat_outer_dims<To>();

    for (int i = 0; i < flat_output.dimension(1); ++i) {
      SymmetricCtVariant<From> const* ct_var =
          std::move(flat_input(i).get<SymmetricCtVariant<From>>());
      OP_REQUIRES(op_ctx, ct_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  "did not unwrap successfully."));
      CtClass const& ct = ct_var->ct;

      // Descrypt() returns coefficients in underlying (e.g. uint64) form after
      // doing the outter modulo and inverse NTT.
      // The current version of the shell library requires the plaintext modulus
      // is 2^(log_t) + 1. The library does the mod 1 as part of decyption.
      OP_REQUIRES_VALUE(std::vector<From> raw_intt_p_mod_t, op_ctx,
                        rlwe::Decrypt<ModularInt>(*key, ct));

      // First need to go back into montgomery ints and Polynomial form mod t.
      // There are two ways to do this, with type punning or by re-allocating.
      // The type punning is more efficient, but is "smellier" and brittle.
      std::vector<ModularInt>& intt_p_mod_t =
          reinterpret_cast<std::vector<ModularInt>&>(raw_intt_p_mod_t);
      // std::vector<ModularInt> intt_p_mod_t(num_slots,
      //                                      ModularInt::ImportZero(pt_params));
      // for (size_t slot = 0; slot < num_slots; ++slot) {
      //   OP_REQUIRES_VALUE(
      //       intt_p_mod_t[slot], op_ctx,
      //       ModularInt::ImportInt(raw_intt_p_mod_t[slot], pt_params));
      // }

      // Finally, compute the NTT to go pack to plaintext
      Polynomial p_mod_t =
          Polynomial::ConvertToNtt(intt_p_mod_t, pt_ntt_params, pt_params);

      // Convert out of montgomery form and store in output tensor.
      for (size_t slot = 0; slot < num_slots; ++slot) {
        To res = p_mod_t.Coeffs()[slot].ExportInt(pt_params);
        res = fix_sign_extend(res, shell_ctx->GetLogT());
        flat_output(slot, i) = res;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KeyGen64").Device(DEVICE_CPU), KeyGenOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("Encrypt64").Device(DEVICE_CPU),
                        EncryptOp<uint64>);

REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<uint8>("dtype"),
    DecryptOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    DecryptOp<uint64, int8>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int16>("dtype"),
    DecryptOp<uint64, int16>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"),
    DecryptOp<uint64, int32>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"),
    DecryptOp<uint64, int64>);
