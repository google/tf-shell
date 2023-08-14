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
class MulCtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit MulCtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

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

      OP_REQUIRES_VALUE(CtClass ct_c, op_ctx, ct_a * ct_b);

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T>
class MulCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit MulCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

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

      OP_REQUIRES_VALUE(CtClass ct_c, op_ctx,
                        ct_a * pt_b);  // shell aborb operation

      SymmetricCtVariant ct_c_var(std::move(ct_c));
      flat_output(i) = std::move(ct_c_var);
    }
  }
};

template <typename T>
class MulPtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;

 public:
  explicit MulPtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

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

      OP_REQUIRES_VALUE(Polynomial pt_c, op_ctx,
                        pt_a.Mul(pt_b, shell_ctx->GetModulusParams()));

      PolynomialVariant<T> pt_c_var(std::move(pt_c));
      flat_output(i) = std::move(pt_c_var);
    }
  }
};

template <typename T, typename PtT>
class MatMulCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;

 public:
  explicit MatMulCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  // This computes the following:
  // def poor_mans_matmul_ct_pt(x, y):
  //  xm, xn = x.shape
  //  ym, yn = y.shape
  //
  //  result = tf.zeros([xm, yn])
  //
  //  for i in range(len(yn)):
  //    result_column_i = tf.zeros([xm, 1])
  //    for j in range(len(ym)):
  //      result_column_i += x_polynomial_column[j] * y_scalar[j, i]
  //
  // TODO(jchoncholas): Support batching. a's dimension may be greater
  // than 2 and matmul is performed for each of the outter dims.

  void Compute(OpKernelContext* op_ctx) override {
    Tensor const& a = op_ctx->input(0);
    Tensor const& b = op_ctx->input(1);

    // a is a vector of Polynomials so first dimension is the number of slots.
    OP_REQUIRES(op_ctx, a.dims() == 1 && b.dims() == 2,
                InvalidArgument("Inputs must have dimension 2."));

    OP_REQUIRES(op_ctx, a.dim_size(a.dims() - 1) == b.dim_size(0),
                InvalidArgument(
                    "Inputs dimensions do not support matrix multiplication."));

    // output is a vector of Polynomials and the first dimension is the number
    // of slots.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(
                               0, TensorShape{b.dim_size(1)}, &output));

    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat_outer_dims<PtT>();
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < b.dim_size(1); ++i) {
      SymmetricCtVariant<T> const* ct_a_var =
          std::move(flat_a(0).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: 0",
                                  " for input a did not unwrap successfully."));
      CtClass const& ct_a = ct_a_var->ct;

      OP_REQUIRES_VALUE(
          auto mint_b, op_ctx,
          ModularInt::ImportInt(flat_b(0, i), ct_a.ModulusParams()));

      OP_REQUIRES_VALUE(CtClass ct_result, op_ctx,
                        ct_a * mint_b);  // Ct * scalar

      for (int j = 1; j < b.dim_size(0); ++j) {
        SymmetricCtVariant<T> const* ct_a_var =
            std::move(flat_a(j).get<SymmetricCtVariant<T>>());
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index:", j,
                            " for input a did not unwrap successfully."));
        CtClass const& ct_a = ct_a_var->ct;

        OP_REQUIRES_VALUE(
            auto mint_b, op_ctx,
            ModularInt::ImportInt(flat_b(j, i), ct_a.ModulusParams()));
        OP_REQUIRES_VALUE(CtClass scaled, op_ctx,
                          ct_a * mint_b);  // Ct * scalar
        OP_REQUIRES_OK(op_ctx, ct_result.AddInPlace(scaled));
      }

      SymmetricCtVariant<T> ct_result_var(std::move(ct_result));
      flat_output(i) = std::move(ct_result_var);
    }
  }
};

template <typename PtT, typename T>
class MatMulPtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using KeyClass = rlwe::SymmetricRlweKey<ModularInt>;

 public:
  explicit MatMulPtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  // TODO(jchoncholas): This is a stub for now. It decryptes, computes the
  // matrix multiplication, and then re-encrypts. Right now shell does not
  // support ciphertext slot rotation via galois keys which is required to
  // perform the addition steps of the matrix multiplication.
  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES(op_ctx, false,
                InvalidArgument("Plaintext * Ciphertext not implemented"));
  }
};

REGISTER_KERNEL_BUILDER(Name("MulCtCt64").Device(DEVICE_CPU),
                        MulCtCtOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("MulCtPt64").Device(DEVICE_CPU),
                        MulCtPtOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("MulPtPt64").Device(DEVICE_CPU),
                        MulPtPtOp<uint64>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<uint8>("dtype"),
    MatMulCtPtOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    MatMulCtPtOp<uint64, int8>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int16>("dtype"),
    MatMulCtPtOp<uint64, int16>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"),
    MatMulCtPtOp<uint64, int32>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"),
    MatMulCtPtOp<uint64, int64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<float>("dtype"),
    MatMulCtPtOp<uint64, float>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<double>("dtype"),
    MatMulCtPtOp<uint64, double>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<uint8>("dtype"),
    MatMulPtCtOp<uint8, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    MatMulPtCtOp<int8, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int16>("dtype"),
    MatMulPtCtOp<int16, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"),
    MatMulPtCtOp<int32, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"),
    MatMulPtCtOp<int64, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<float>("dtype"),
    MatMulPtCtOp<float, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<double>("dtype"),
    MatMulPtCtOp<double, uint64>);