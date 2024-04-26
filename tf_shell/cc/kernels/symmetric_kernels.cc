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
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bgv_ciphertext.h"
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
using tensorflow::errors::InvalidArgument;

template <typename T>
class KeyGenOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;

 public:
  explicit KeyGenOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the context variant from the input which holds all the shell objects
    // needed to sample a secret key.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Prng* prng = shell_ctx_var->prng_.get();

    // Allocate output tensor which is a scalar holding the secret key.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));

    OP_REQUIRES_VALUE(
        Key k, op_ctx,
        Key::Sample(shell_ctx->LogN(), shell_ctx_var->noise_variance_,
                    shell_ctx->MainPrimeModuli(), prng));

    SymmetricKeyVariant<T> key_variant(std::move(k));
    out->scalar<Variant>()() = std::move(key_variant);
  }
};

template <typename T>
class EncryptOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Polynomial = rlwe::RnsPolynomial<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit EncryptOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    size_t num_slots = 1 << shell_ctx->LogN();

    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 1));
    Key const* secret_key = &secret_key_var->key;

    Tensor const& input = op_ctx->input(2);

    // Allocate the output tensor which is the same shape as the input
    // tensor holding the shell polynomials.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_input = input.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    auto enc_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        PolynomialVariant<T> const* pv =
            std::move(flat_input(i).get<PolynomialVariant<T>>());
        OP_REQUIRES(op_ctx, pv != nullptr,
                    InvalidArgument("PolynomialVariant at flat index:", i,
                                    "did not unwrap successfully."));
        Polynomial const& p = pv->poly;

        SymmetricCt ciphertext = secret_key
                                     ->template EncryptPolynomialBgv<Encoder>(
                                         p, shell_ctx_var->encoder_.get(),
                                         shell_ctx_var->error_params_.get(),
                                         shell_ctx_var->prng_.get())
                                     .value();

        SymmetricCtVariant ciphertext_var(std::move(ciphertext));
        flat_output(i) = std::move(ciphertext_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_enc = 2200 * num_slots;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_enc,
                             enc_in_range);
  }
};

template <typename From, typename To>
class DecryptOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<From>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit DecryptOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<From> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<From>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Encoder const* encoder = shell_ctx_var->encoder_.get();

    OP_REQUIRES_VALUE(SymmetricKeyVariant<From> const* key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<From>>(op_ctx, 1));
    Key const* secret_key = &key_var->key;

    Tensor const& input = op_ctx->input(2);
    OP_REQUIRES(op_ctx, input.dim_size(0) > 0,
                InvalidArgument("Cannot decrypt empty ciphertext"));
    auto flat_input = input.flat<Variant>();

    size_t num_slots = 1 << shell_ctx->LogN();

    // Allocate the output tensor so that the shape include an extra dimension
    // at the beginning to hold values in the slots unpacked from the
    // ciphertext.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.InsertDimWithStatus(0, num_slots));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_output = output->flat_outer_dims<To>();

    auto dec_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<From> const* ct_var =
            std::move(flat_input(i).get<SymmetricCtVariant<From>>());
        OP_REQUIRES(op_ctx, ct_var != nullptr,
                    InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                    " did not unwrap successfully."));
        SymmetricCt const& ct = ct_var->ct;

        // Decrypt() returns coefficients in underlying (e.g. uint64) form after
        // doing the outer modulo and inverse NTT.
        OP_REQUIRES_VALUE(
            std::vector<From> decryptions, op_ctx,
            secret_key->template DecryptBgv<Encoder>(ct, encoder));

        if constexpr (std::is_signed<To>::value) {
          // Map the plaintext modulus field back into signed integers.
          // Effectively switches the modulus from t to 2^(num bits in `From`)
          // handling the sign bits appropriately.
          OP_REQUIRES_VALUE(std::vector<std::make_signed_t<To>> nums, op_ctx,
                            encoder->template UnwrapToSigned<To>(decryptions));

          for (size_t slot = 0; slot < num_slots; ++slot) {
            flat_output(slot, i) = static_cast<To>(nums[slot]);
          }
        } else {
          // both `From` and `To` are unsigned, just cast and copy.
          for (size_t slot = 0; slot < num_slots; ++slot) {
            flat_output(slot, i) = static_cast<To>(decryptions[slot]);
          }
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_dec = 0.12f * num_slots;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(1), cost_per_dec,
                             dec_in_range);
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
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<uint16>("dtype"),
    DecryptOp<uint64, uint16>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int16>("dtype"),
    DecryptOp<uint64, int16>);

REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<uint32>("dtype"),
    DecryptOp<uint64, uint32>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"),
    DecryptOp<uint64, int32>);

REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<uint64>("dtype"),
    DecryptOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("Decrypt64").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"),
    DecryptOp<uint64, int64>);
