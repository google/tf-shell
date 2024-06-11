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
#include "shell_encryption/rns/rns_modulus.h"
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

// The TensorFlow custom Ops below use the same class for addition and
// subtraction. To do so, they are templated on a functor that performs either
// the addition or subtraction. These functors are defined below.

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

template <typename LhsT, typename RhsT>
struct ShellAddInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs) const {
    return lhs.AddInPlace(rhs);
  }
};

template <typename LhsT, typename RhsT>
struct ShellSubInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs) const {
    return lhs.SubInPlace(rhs);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellAddWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const& p) const {
    return lhs.Add(rhs, p);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellSubWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const& p) const {
    return lhs.Sub(rhs, p);
  }
};

template <typename T, typename ShellAddSub>
class AddCtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit AddCtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments. The 0th argument is the context, which is not
    // directly used in this op but required for graph optimization.
    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = MyBFlat<Variant>(op_ctx, a, bcast.x_reshape(), bcast.x_bcast());
    auto flat_b = MyBFlat<Variant>(op_ctx, b, bcast.y_reshape(), bcast.y_bcast());

    // Check the inputs have the same shape.
    OP_REQUIRES(
        op_ctx, flat_a.size() == flat_b.size(),
        InvalidArgument("Broadcasted inputs must have the same shape."));

    // Allocate the output tensor which is the same size as one of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      SymmetricCt const& ct_a = ct_a_var->ct;

      SymmetricCtVariant<T> const* ct_b_var =
          std::move(flat_b(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_b_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input b did not unwrap successfully."));
      SymmetricCt const& ct_b = ct_b_var->ct;

      ShellAddSub add_or_sub;
      OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx, add_or_sub(ct_a, ct_b));

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T, typename ShellAddSub>
class AddCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit AddCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments. The 0th argument is the context, which is not
    // directly used in this op but required for graph optimization.
    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = MyBFlat<Variant>(op_ctx, a, bcast.x_reshape(), bcast.x_bcast());
    auto flat_b = MyBFlat<Variant>(op_ctx, b, bcast.y_reshape(), bcast.y_bcast());

    // Check the inputs have the same shape.
    OP_REQUIRES(
        op_ctx, flat_a.size() == flat_b.size(),
        InvalidArgument("Broadcasted inputs must have the same shape."));

    // Allocate the output tensor which is the same size as one of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      SymmetricCt const& ct_a = ct_a_var->ct;

      PolynomialVariant<T> const* pv_b_var =
          std::move(flat_b(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_b_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index: ", i,
                                  " for input b did not unwrap successfully."));
      RnsPolynomial const& pt_b = pv_b_var->poly;

      ShellAddSub add_or_sub;
      OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx, add_or_sub(ct_a, pt_b));

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T, typename ShellAddSubWithParams>
class AddPtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit AddPtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = MyBFlat<Variant>(op_ctx, a, bcast.x_reshape(), bcast.x_bcast());
    auto flat_b = MyBFlat<Variant>(op_ctx, b, bcast.y_reshape(), bcast.y_bcast());

    // Check the inputs have the same shape.
    OP_REQUIRES(
        op_ctx, flat_a.size() == flat_b.size(),
        InvalidArgument("Broadcasted inputs must have the same shape."));

    // Allocate the output tensor which is the same size as one of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pv_a_var =
          std::move(flat_a(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_a_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      RnsPolynomial const& pt_a = pv_a_var->poly;

      PolynomialVariant<T> const* pv_b_var =
          std::move(flat_b(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pv_b_var != nullptr,
                  InvalidArgument("PolynomialVariant at flat index: ", i,
                                  " for input b did not unwrap successfully."));
      RnsPolynomial const& pt_b = pv_b_var->poly;

      ShellAddSubWithParams add_or_sub;
      OP_REQUIRES_VALUE(RnsPolynomial pt_c, op_ctx,
                        add_or_sub(pt_a, pt_b, shell_ctx->MainPrimeModuli()));

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
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit NegCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments. The 0th argument is the context, which is not
    // directly used in this op but required for graph optimization.
    Tensor const& a = op_ctx->input(1);

    // Allocate the output tensor which is the same size as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    // Set up flat views of the input and output tensors.
    auto flat_a = a.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      SymmetricCt const& ct_a = ct_a_var->ct;

      OP_REQUIRES_VALUE(auto ct_out, op_ctx, ct_a.Negate());

      SymmetricCtVariant ct_out_var(std::move(ct_out));
      flat_output(i) = std::move(ct_out_var);
    }
  }
};

template <typename T>
class NegPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit NegPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);

    // Allocate the output tensor which is the same size as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    // Set up flat views of the input and output tensors.
    auto flat_a = a.flat<Variant>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      PolynomialVariant<T> const* pt_a_var =
          std::move(flat_a(i).get<PolynomialVariant<T>>());
      OP_REQUIRES(op_ctx, pt_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      RnsPolynomial const& pt_a = pt_a_var->poly;

      OP_REQUIRES_VALUE(RnsPolynomial pt_out, op_ctx,
                        pt_a.Negate(shell_ctx->MainPrimeModuli()));
      PolynomialVariant pt_out_var(std::move(pt_out));
      flat_output(i) = std::move(pt_out_var);
    }
  }
};

using BaseInt64 = uint64;
using ModularInt64 = rlwe::MontgomeryInt<BaseInt64>;
// using ModularIntParams64 = typename rlwe::MontgomeryInt<BaseInt64>::Params;
using Moduli = absl::Span<rlwe::PrimeModulus<ModularInt64> const* const>;
using RnsPolynomial64 = rlwe::RnsPolynomial<ModularInt64>;
using SymmetricCt64 = rlwe::RnsBgvCiphertext<ModularInt64>;

REGISTER_KERNEL_BUILDER(
    Name("AddCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64,
              ShellAddFunctor<SymmetricCt64, SymmetricCt64, SymmetricCt64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64,
              ShellAddFunctor<SymmetricCt64, SymmetricCt64, RnsPolynomial64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellAddWithParamsFunctor<RnsPolynomial64, RnsPolynomial64,
                                        RnsPolynomial64, Moduli>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64,
              ShellSubFunctor<SymmetricCt64, SymmetricCt64, SymmetricCt64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64,
              ShellSubFunctor<SymmetricCt64, SymmetricCt64, RnsPolynomial64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellSubWithParamsFunctor<RnsPolynomial64, RnsPolynomial64,
                                        RnsPolynomial64, Moduli>>);

REGISTER_KERNEL_BUILDER(Name("NegCt64").Device(DEVICE_CPU), NegCtOp<BaseInt64>);
REGISTER_KERNEL_BUILDER(Name("NegPt64").Device(DEVICE_CPU), NegPtOp<BaseInt64>);
