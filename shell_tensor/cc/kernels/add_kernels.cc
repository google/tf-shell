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

#include "absl/status/status.h"
#include "context_variant.h"
#include "polynomial_variant.h"
#include "prng_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
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
using tensorflow::uint64;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

template <typename OutT, typename LhsT, typename RhsT>
struct ShellAddFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs,
                                            RhsT const& rhs) const {
    return lhs + rhs;
  }
};

template <typename OutT, typename LhsT, typename RhsT>
struct ShellSubFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs,
                                            RhsT const& rhs) const {
    return lhs - rhs;
  }
};

template <typename LhsT, typename RhsT, typename ModParamsT>
struct ShellAddInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs,
                                    ModParamsT const* p) const {
    return lhs.AddInPlace(rhs, p);
  }
};

template <typename LhsT, typename RhsT, typename ModParamsT>
struct ShellSubInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs,
                                    ModParamsT const* p) const {
    return lhs.SubInPlace(rhs, p);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellAddWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const* p) const {
    return lhs.Add(rhs, p);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellSubWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const* p) const {
    return lhs.Sub(rhs, p);
  }
};

template <typename T, typename ShellAddSub>
class AddCtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit AddCtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    Tensor const& a = op_ctx->input(0);
    Tensor const& b = op_ctx->input(1);

    OP_REQUIRES(op_ctx, a.shape() == b.shape(),
                InvalidArgument("Inputs must have the same shape."));

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " for input a did not unwrap successfully."));
      CtClass const& ct_a = ct_a_var->ct;

      SymmetricCtVariant<T> const* ct_b_var =
          std::move(flat_b(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_b_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " for input b did not unwrap successfully."));
      CtClass const& ct_b = ct_b_var->ct;

      ShellAddSub add_or_sub;
      OP_REQUIRES_VALUE(CtClass ct_c, op_ctx, add_or_sub(ct_a, ct_b));

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T, typename ShellAddSubInPlace>
class AddCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit AddCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    Tensor const& a = op_ctx->input(0);
    Tensor const& b = op_ctx->input(1);

    OP_REQUIRES(op_ctx, a.shape() == b.shape(),
                InvalidArgument("Inputs must have the same shape."));

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " for input a did not unwrap successfully."));
      CtClass const& ct_a = ct_a_var->ct;

      PolynomialVariant<T> const* pv_b_var =
          std::move(flat_b(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_b_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  " for input b did not unwrap successfully."));
      Polynomial const& pt_b = pv_b_var->poly;

      // I'd like to just do the below but I think this copies ciphertext
      // Component 0 instead of modifying in place and wont work.
      //   CtClass ct_c = ct_a; // deep copy
      //   OP_REQUIRES_OK(op_ctx, ct_c.Component(0).AddInPlace(pt_b,
      //                  ct_a.ModulusParams()));
      //
      // So instead I do the below:
      std::vector<Polynomial> raw_ct(ct_a.Len());

      // Add or subtract pt to ct component 0.
      OP_REQUIRES_VALUE(Polynomial ct_a_comp, op_ctx,
                        ct_a.Component(0));  // deep copy

      // Perform the + or - operation with the template functor
      ShellAddSubInPlace add_or_sub_in_place;
      OP_REQUIRES_OK(
          op_ctx, add_or_sub_in_place(ct_a_comp, pt_b, ct_a.ModulusParams()));
      raw_ct[0] = std::move(ct_a_comp);

      // Copy the rest of the ct components.
      for (size_t comp_i = 1; comp_i < ct_a.Len(); ++comp_i) {
        OP_REQUIRES_VALUE(Polynomial ct_a_comp, op_ctx,
                          ct_a.Component(comp_i));  // deep copy
        raw_ct[comp_i] = std::move(ct_a_comp);
      }

      // Create result ciphertext from components.
      CtClass ct_c(std::move(raw_ct), ct_a.PowerOfS(),
                   ct_a.Error() + ct_a.ErrorParams()->B_plaintext(),
                   ct_a.ModulusParams(), ct_a.ErrorParams());

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T, typename ShellAddSubWithParams>
class AddPtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;

 public:
  explicit AddPtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<T>>(op_ctx, 0, &shell_ctx_var));
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    OP_REQUIRES(op_ctx, a.shape() == b.shape(),
                InvalidArgument("Inputs must have the same shape."));

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pv_a_var =
          std::move(flat_a(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_a_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  " for input a did not unwrap successfully."));
      Polynomial const& pt_a = pv_a_var->poly;

      PolynomialVariant<T> const* pv_b_var =
          std::move(flat_b(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_b_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index:", i,
                                  " for input b did not unwrap successfully."));
      Polynomial const& pt_b = pv_b_var->poly;

      ShellAddSubWithParams add_or_sub;
      OP_REQUIRES_VALUE(Polynomial pt_c, op_ctx,
                        add_or_sub(pt_a, pt_b, shell_ctx->GetModulusParams()));

      PolynomialVariant<T> pt_c_var(std::move(pt_c));
      flat_output(i) = std::move(pt_c_var);
    }
  }
};

template <typename T>
class NegCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename ModularInt::Params;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit NegCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<T>>(op_ctx, 0, &shell_ctx_var));
    ModularIntParams const* pt_params = shell_ctx_var->pt_params.get();
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    Tensor const& a = op_ctx->input(1);

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    auto flat_a = a.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " for input a did not unwrap successfully."));
      CtClass const& ct_a = ct_a_var->ct;

      // Recreate the ciphertext by negating each component.
      std::vector<Polynomial> raw_ct(ct_a.Len());

      // Negate each ct component.
      for (size_t comp_i = 0; comp_i < ct_a.Len(); ++comp_i) {
        OP_REQUIRES_VALUE(Polynomial ct_a_comp, op_ctx,
                          ct_a.Component(comp_i));  // deep copy
        raw_ct[comp_i] =
            std::move(ct_a_comp.NegateInPlace(ct_a.ModulusParams()));
      }

      // Create result ciphertext from components.
      CtClass ct_out(std::move(raw_ct), ct_a.PowerOfS(),
                     ct_a.Error() + ct_a.ErrorParams()->B_plaintext(),
                     ct_a.ModulusParams(), ct_a.ErrorParams());

      SymmetricCtVariant ct_out_var(std::move(ct_out));
      flat_output(i) = std::move(ct_out_var);
    }
  }
};

template <typename T>
class NegPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;

 public:
  explicit NegPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    ContextVariant<T> const* shell_ctx_var;
    OP_REQUIRES_OK(op_ctx,
                   GetVariant<ContextVariant<T>>(op_ctx, 0, &shell_ctx_var));
    Context const* shell_ctx = shell_ctx_var->ct_context.get();

    Tensor const& a = op_ctx->input(1);

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    auto flat_a = a.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pt_a_var =
          std::move(flat_a(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pt_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index:", i,
                                  " for input a did not unwrap successfully."));
      Polynomial const& pt_a = pt_a_var->poly;

      Polynomial pt_out = pt_a.Negate(shell_ctx->GetModulusParams());
      PolynomialVariant pt_out_var(std::move(pt_out));
      flat_output(i) = std::move(pt_out_var);
    }
  }
};

using BaseInt64 = uint64;
using ModularInt64 = rlwe::MontgomeryInt<BaseInt64>;
using ModularIntParams64 = typename rlwe::MontgomeryInt<BaseInt64>::Params;
using Polynomial64 = rlwe::Polynomial<ModularInt64>;
using CtClass64 = rlwe::SymmetricRlweCiphertext<ModularInt64>;

REGISTER_KERNEL_BUILDER(
    Name("AddCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64, ShellAddFunctor<CtClass64, CtClass64, CtClass64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64, ShellAddInPlaceFunctor<Polynomial64, Polynomial64,
                                                ModularIntParams64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellAddWithParamsFunctor<Polynomial64, Polynomial64,
                                        Polynomial64, ModularIntParams64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64, ShellSubFunctor<CtClass64, CtClass64, CtClass64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64, ShellSubInPlaceFunctor<Polynomial64, Polynomial64,
                                                ModularIntParams64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellSubWithParamsFunctor<Polynomial64, Polynomial64,
                                        Polynomial64, ModularIntParams64>>);

REGISTER_KERNEL_BUILDER(Name("NegCt64").Device(DEVICE_CPU), NegCtOp<BaseInt64>);
REGISTER_KERNEL_BUILDER(Name("NegPt64").Device(DEVICE_CPU), NegPtOp<BaseInt64>);