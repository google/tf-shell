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
#include "shell_encryption/montgomery.h"
#include "symmetric_variants.h"
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
using tensorflow::uint64;
using tensorflow::Variant;

template <typename T>
class ModulusReduceKeyOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;

 public:
  explicit ModulusReduceKeyOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 0));
    Key secret_key = secret_key_var->key;  // Deep copy.

    // Allocate a scalar output tensor to store the reduced key.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));

    OP_REQUIRES_OK(op_ctx, secret_key.ModReduce());

    // Store the reduced key in the output tensor.
    SymmetricKeyVariant<T> reduced_key_variant(std::move(secret_key));
    out->scalar<Variant>()() = std::move(reduced_key_variant);
  }
};

template <typename T>
class ModulusReduceCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit ModulusReduceCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    OP_REQUIRES(op_ctx, a.dim_size(0) > 0,
                InvalidArgument("Cannot modulus reduce an empty ciphertext."));
    auto flat_a = a.flat<Variant>();

    OP_REQUIRES_VALUE(bool preserve_plaintext, op_ctx,
                      GetScalar<bool>(op_ctx, 2));

    // Allocate the output tensor.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Get the first ciphertext from the input so we can determine some of the
    // parameters about the ciphertexts in this tensor.
    SymmetricCtVariant<T> const* first_a_var =
        std::move(flat_a(0).get<SymmetricCtVariant<T>>());
    OP_REQUIRES(op_ctx, first_a_var != nullptr,
                InvalidArgument(
                    "First SymmetricCtVariant did not unwrap successfully."));
    SymmetricCt first_a = first_a_var->ct;  // Deep copy. ModReduce is in place.

    // Gather the parameters for the modulus reduction.
    size_t level = first_a.Level();
    auto q_inv_mod_qs = shell_ctx->MainPrimeModulusInverseResidues();
    OP_REQUIRES(
        op_ctx, level < q_inv_mod_qs.size(),
        InvalidArgument(
            "Ciphertext level does not match num inverse prime moduli."));
    OP_REQUIRES(op_ctx, level < q_inv_mod_qs[level].zs.size(),
                InvalidArgument("Ciphertext level does not match rns size."));
    auto ql_inv = q_inv_mod_qs[level].Prefix(level);

    auto t = shell_ctx->PlaintextModulus();

    // If all the ciphertexts in the tensor have been encrypted with the same
    // secret key, their moduli should all be pointers to the moduli in the
    // secret key. Copy their pointers here.
    std::vector<rlwe::PrimeModulus<ModularInt> const*> moduli;
    for (rlwe::PrimeModulus<ModularInt> const* modulus : first_a.Moduli()) {
      moduli.push_back(modulus);
    }
    auto reduced_moduli = moduli;
    reduced_moduli.pop_back();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      // Get the ciphertext wrapper from the input.
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " did not unwrap successfully."));

      if (preserve_plaintext) {
        SymmetricCt result_ct =
            ct_a_var->ct;  // Deep copy. ModReduce is in place.
        OP_REQUIRES_OK(op_ctx, result_ct.ModReduce(t, ql_inv));

        // Store in the output.
        SymmetricCtVariant<T> result_var(std::move(result_ct));
        flat_output(i) = std::move(result_var);
      } else {
        SymmetricCt const& ct_a =
            ct_a_var->ct;  // Just get a reference. Copy is below.
        // SymmetricCt ct_a = ct_a_var->ct;  // Deep copy. ModReduce is in
        // place.

        // ModReduceMsb on each of the ciphertext components will result in an
        // encryption of the plaintext divided by the last moduli, q_l after
        // rounding. SymmetricCt ModReduce uses the Lsb version so we operate on
        // the components individually to use the Msb version.
        std::vector<RnsPolynomial> reduced_components;

        for (int i = 0; i < ct_a.Len(); ++i) {
          OP_REQUIRES_VALUE(RnsPolynomial c, op_ctx,
                            ct_a.Component(i));  // Here is the copy

          OP_REQUIRES_OK(op_ctx, c.ModReduceMsb(ql_inv, moduli));

          reduced_components.push_back(std::move(c));
        }

        // Create a new ciphertext with the reduced components.
        SymmetricCt result_ct(std::move(reduced_components), reduced_moduli,
                              ct_a.PowerOfS(), ct_a.Error(),
                              ct_a.ErrorParams());

        // Store in the output.
        SymmetricCtVariant<T> result_var(std::move(result_ct));
        flat_output(i) = std::move(result_var);
      }
    }
  }
};

template <typename T>
class ModulusReducePtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit ModulusReducePtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    OP_REQUIRES(op_ctx, a.dim_size(0) > 0,
                InvalidArgument("Cannot modulus reduce an empty ciphertext."));
    auto flat_a = a.flat<Variant>();

    OP_REQUIRES_VALUE(bool preserve_plaintext, op_ctx,
                      GetScalar<bool>(op_ctx, 2));

    // Allocate the output tensor.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Get the first ciphertext from the input so we can determine some of the
    // parameters about the ciphertexts in this tensor.
    PolynomialVariant<T> const* pt_a_var =
        std::move(flat_a(0).get<PolynomialVariant<T>>());
    OP_REQUIRES(op_ctx, pt_a_var != nullptr,
                InvalidArgument(
                    "First PolynomialVariant did not unwrap successfully."));

    // Gather the parameters for the modulus reduction.
    size_t level = pt_a_var->poly.NumModuli() - 1;
    auto q_inv_mod_qs = shell_ctx->MainPrimeModulusInverseResidues();
    OP_REQUIRES(
        op_ctx, level < q_inv_mod_qs.size(),
        InvalidArgument(
            "Polynomial level does not match num inverse prime moduli."));
    OP_REQUIRES(op_ctx, level < q_inv_mod_qs[level].zs.size(),
                InvalidArgument("Polynomial level does not match rns size."));
    auto ql_inv = q_inv_mod_qs[level].Prefix(level);
    auto main_moduli = shell_ctx->MainPrimeModuli();
    auto t = shell_ctx->PlaintextModulus();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pt_a_var =
          std::move(flat_a(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pt_a_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  " did not unwrap successfully."));

      // Deep copy the polynomial because ModReduce is in place.
      OP_REQUIRES_VALUE(RnsPolynomial pt_a, op_ctx,
                        RnsPolynomial::Create(pt_a_var->poly.Coeffs(),
                                              pt_a_var->poly.IsNttForm()));

      if (preserve_plaintext) {
        OP_REQUIRES_OK(op_ctx, pt_a.ModReduceLsb(t, ql_inv, main_moduli));
      } else {
        OP_REQUIRES_OK(op_ctx, pt_a.ModReduceMsb(ql_inv, main_moduli));
      }

      PolynomialVariant<T> result_var(std::move(pt_a));
      flat_output(i) = std::move(result_var);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ModulusReduceKey64").Device(DEVICE_CPU),
                        ModulusReduceKeyOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ModulusReduceCt64").Device(DEVICE_CPU),
                        ModulusReduceCtOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ModulusReducePt64").Device(DEVICE_CPU),
                        ModulusReducePtOp<uint64>);
