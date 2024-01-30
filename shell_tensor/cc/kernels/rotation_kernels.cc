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
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/rns/rns_galois_key.h"
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

constexpr int kLogGadgetBase = 4;
constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;

template <typename T>
class RotationKeyGenOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using Gadget = rlwe::RnsGadget<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using PowerAndKey = typename RotationKeyVariant<T>::PowerAndKey;

 public:
  explicit RotationKeyGenOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 1));
    Key const* secret_key = &secret_key_var->key;

    // Create the gadget.
    int level = shell_ctx->NumMainPrimeModuli() - 1;
    OP_REQUIRES_VALUE(auto q_hats, op_ctx,
                      shell_ctx->MainPrimeModulusComplements(level));
    OP_REQUIRES_VALUE(auto q_hat_invs, op_ctx,
                      shell_ctx->MainPrimeModulusCrtFactors(level));
    std::vector<size_t> log_bs(shell_ctx->NumMainPrimeModuli(), kLogGadgetBase);
    OP_REQUIRES_VALUE(Gadget raw_gadget, op_ctx,
                      Gadget::Create(shell_ctx->LogN(), log_bs, q_hats,
                                     q_hat_invs, shell_ctx->MainPrimeModuli()));

    // Store the gadget in a variant.
    // Once it has been std::moved into it's final memory location, it can
    // be used to create the rotation keys.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));
    RotationKeyVariant<T> v_out(std::move(raw_gadget));
    out->scalar<Variant>()() = std::move(v_out);
    RotationKeyVariant<T>* key_variant =
        out->scalar<Variant>()(0).get<RotationKeyVariant<T>>();
    OP_REQUIRES(op_ctx, key_variant != nullptr,
                InvalidArgument(
                    "RotationKeyVariant did not unwrap successfully. Saw: '",
                    out->scalar<Variant>()().DebugString(), "'"));
    Gadget* gadget = &key_variant->gadget;

    // The substitution power for Galois rotation by one slot.
    constexpr int base_power = 5;
    int sub_power = base_power;

    // This method of rotation only allows us to rotate within half of the
    // polynomial slots. E.g. for n slots, slot 0 can be rotated to at most
    // n/2-1 and n/2 to n-1. This has implications for how batching is done if
    // performing backpropagation under encryption.
    int num_rotation_keys = 1 << (shell_ctx->LogN() - 1);
    int two_n = 1 << (shell_ctx->LogN() + 1);

    // TODO: baby-step giant-step.
    for (int i = 1; i < num_rotation_keys; ++i) {
      OP_REQUIRES_VALUE(RotationKey k, op_ctx,
                        RotationKey::CreateForBgv(
                            *secret_key, sub_power, secret_key->Variance(),
                            gadget, shell_ctx->PlaintextModulus(), kPrngType));
      PowerAndKey p_and_k{sub_power, std::move(k)};
      key_variant->keys.emplace(i, std::move(p_and_k));
      sub_power *= base_power;
      sub_power %= two_n;
    }
  }
};

template <typename T>
class RollOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using PowerAndKey = typename RotationKeyVariant<T>::PowerAndKey;

 public:
  explicit RollOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES_VALUE(RotationKeyVariant<T> const* rotation_key_var, op_ctx,
                      GetVariant<RotationKeyVariant<T>>(op_ctx, 0));
    OP_REQUIRES(
        op_ctx, rotation_key_var != nullptr,
        InvalidArgument("RotationKeyVariant a did not unwrap successfully."));
    std::map<int, PowerAndKey> const* keys = &rotation_key_var->keys;

    Tensor const& value = op_ctx->input(1);
    OP_REQUIRES_VALUE(int64 shift, op_ctx, GetScalar<int64>(op_ctx, 2));
    shift = -shift;  // tensorflow.roll() uses negative shift for left shift.

    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot roll empty ciphertext."));

    auto flat_value = value.flat<Variant>();

    // Setup the output.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, value.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Recover num_slots from first ciphertext to validate shift argument.
    SymmetricCtVariant<T> const* ct_var =
        std::move(flat_value(0).get<SymmetricCtVariant<T>>());
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();

    OP_REQUIRES(op_ctx, abs(shift) < num_slots / 2,
                InvalidArgument("Shifting by too many slots, shift of '", shift,
                                "' must be less than '", num_slots / 2, "'"));

    // Handle negative shift.
    // Careful with c++ modulo operator on negative numbers.
    if (shift < 0) {
      shift += num_slots / 2;
    }

    PowerAndKey const* p_and_k;
    if (shift != 0) {
      OP_REQUIRES(op_ctx, keys->find(shift) != keys->end(),
                  InvalidArgument("No key for shift of '", shift, "'"));
      p_and_k = &keys->at(shift);
    }

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      SymmetricCtVariant<T> const* ct_var =
          std::move(flat_value(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(op_ctx, ct_var != nullptr,
                  InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                  " for input a did not unwrap successfully."));
      SymmetricCt const& ct = ct_var->ct;

      if (shift == 0) {
        SymmetricCtVariant ct_out_var(ct);
        flat_output(i) = std::move(ct_out_var);
      } else {
        OP_REQUIRES_VALUE(auto ct_sub, op_ctx,
                          ct.Substitute(p_and_k->substitution_power));
        OP_REQUIRES_VALUE(auto ct_rot, op_ctx, p_and_k->key.ApplyTo(ct_sub));

        SymmetricCtVariant ct_out_var(std::move(ct_rot));
        flat_output(i) = std::move(ct_out_var);
      }
    }
  }
};

template <typename T>
class ReduceSumByRotationOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using PowerAndKey = typename RotationKeyVariant<T>::PowerAndKey;

 public:
  explicit ReduceSumByRotationOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Recover the input tensor.
    Tensor const& value = op_ctx->input(0);
    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot reduce_sum an empty ciphertext."));

    auto flat_value = value.flat<Variant>();

    // Recover the input rotation keys.
    OP_REQUIRES_VALUE(RotationKeyVariant<T> const* rotation_key_var, op_ctx,
                      GetVariant<RotationKeyVariant<T>>(op_ctx, 1));
    OP_REQUIRES(
        op_ctx, rotation_key_var != nullptr,
        InvalidArgument("RotationKeyVariant a did not unwrap successfully."));
    std::map<int, PowerAndKey> const* keys = &rotation_key_var->keys;

    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, value.shape(), &output));
    auto flat_output = output->flat<Variant>();

    for (int i = 0; i < flat_output.dimension(0); ++i) {
      // Learn how many slots there are from first ciphertext and create a
      // deep copy to hold the sum.
      SymmetricCtVariant<T> const* ct_var =
          std::move(flat_value(i).get<SymmetricCtVariant<T>>());
      OP_REQUIRES(
          op_ctx, ct_var != nullptr,
          InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
      SymmetricCt sum = ct_var->ct;  // deep copy to start the sum.
      int const num_slots = 1 << sum.LogN();

      // Add the rotations to the sum.
      // Note the ciphertext rotations operate on each half of the ciphertext
      // separately. So the max rotation is by half the number of slots.
      for (int shift = 1; shift < num_slots / 2; shift <<= 1) {
        // TODO if debug
        OP_REQUIRES(op_ctx, keys->find(shift) != keys->end(),
                    InvalidArgument("No key for shift of '", shift, "'"));
        PowerAndKey const& p_and_k = keys->at(shift);

        // Rotate by the shift.
        OP_REQUIRES_VALUE(auto ct_sub, op_ctx,
                          sum.Substitute(p_and_k.substitution_power));
        OP_REQUIRES_VALUE(auto ct_rot, op_ctx, p_and_k.key.ApplyTo(ct_sub));

        // Add to the sum.
        OP_REQUIRES_OK(op_ctx, sum.AddInPlace(ct_rot));
      }

      SymmetricCtVariant ct_out_var(std::move(sum));
      flat_output(i) = std::move(ct_out_var);
    }
  }
};

template <typename T>
class ReduceSumOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using PowerAndKey = typename RotationKeyVariant<T>::PowerAndKey;

 public:
  explicit ReduceSumOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Recover the input tensor.
    Tensor const& value = op_ctx->input(0);
    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot reduce_sum an empty ciphertext."));

    // Recover the axis to reduce over.
    Tensor const& axis_tensor = op_ctx->input(1);
    OP_REQUIRES(op_ctx, axis_tensor.NumElements() == 1,
                InvalidArgument("axis must be scalar, saw shape: ",
                                axis_tensor.shape().DebugString()));
    OP_REQUIRES_VALUE(int64 axis, op_ctx, GetScalar<int64>(op_ctx, 1));

    // The axis to reduce over.
    int dim_to_reduce = axis - 1;

    // Check axis is within dim size.
    OP_REQUIRES(op_ctx, dim_to_reduce < value.dims(),
                InvalidArgument("Cannot reduce_sum over polynomial_axis '",
                                dim_to_reduce, "' (axis '", axis,
                                "') for input with shape ",
                                value.shape().DebugString()));

    uint8_t dim_sz_to_reduce = value.dim_size(dim_to_reduce);

    // Create a temp Tensor to hold intermediate sums during the reduction.
    // It is the same size as the input Tensor.
    Tensor intermediate_sums;
    auto intermediate_sums_shape = value.shape();
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_temp(tensorflow::DT_VARIANT,
                                                 intermediate_sums_shape,
                                                 &intermediate_sums));

    auto flat_intermediate_sums =
        intermediate_sums.flat_inner_outer_dims<Variant>(dim_to_reduce - 1);
    auto flat_value = value.flat_inner_outer_dims<Variant>(dim_to_reduce - 1);

    // Setup the output.
    Tensor* output;
    auto output_shape = value.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.RemoveDimWithStatus(dim_to_reduce));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    // Setup a shape to access the output Tensor as a flat Tensor, with the
    // same indexing as the input Tensor excluding the dimension to reduce.
    int inner_shape = flat_value.dimension(0);
    int outer_shape = flat_value.dimension(2);
    auto flat_output = output->shaped<Variant, 2>({inner_shape, outer_shape});

    // Take the first ciphertext in the chip and add all the other chips to it.
    for (int i = 0; i < flat_output.dimension(0); ++i) {
      for (int j = 0; j < flat_output.dimension(1); ++j) {
        // Get the first chip.
        SymmetricCtVariant<T> const* first_ct_var =
            std::move(flat_value(i, 0, j).get<SymmetricCtVariant<T>>());
        OP_REQUIRES(op_ctx, first_ct_var != nullptr,
                    InvalidArgument(
                        "SymmetricCtVariant a did not unwrap successfully."));
        SymmetricCt sum = first_ct_var->ct;  // deep copy to start the sum.

        // Add the remaining chips.
        for (int chip_dim = 1; chip_dim < dim_sz_to_reduce; ++chip_dim) {
          SymmetricCtVariant<T> const* ct_var = std::move(
              flat_value(i, chip_dim, j).get<SymmetricCtVariant<T>>());
          OP_REQUIRES(op_ctx, ct_var != nullptr,
                      InvalidArgument(
                          "SymmetricCtVariant a did not unwrap successfully."));
          SymmetricCt const& ct = ct_var->ct;

          // Perform the addition.
          OP_REQUIRES_OK(op_ctx, sum.AddInPlace(ct));
        }

        // Store in the output.
        SymmetricCtVariant res_var(std::move(sum));
        flat_output(i, j) = std::move(res_var);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("RotationKeyGen64").Device(DEVICE_CPU),
                        RotationKeyGenOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("Roll64").Device(DEVICE_CPU), RollOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ReduceSumByRotation64").Device(DEVICE_CPU),
                        ReduceSumByRotationOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ReduceSum64").Device(DEVICE_CPU),
                        ReduceSumOp<uint64>);