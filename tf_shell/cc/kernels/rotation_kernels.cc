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

 public:
  explicit RotationKeyGenOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 1));

    std::shared_ptr<Key> const secret_key = secret_key_var->key;

    // Allocate the output tensor which is a scalar containing the rotation key.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));
    // Create the output variant
    RotationKeyVariant<T> v_out;

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

    auto gadget_ptr = std::make_shared<Gadget>(std::move(raw_gadget));
    v_out.gadget = gadget_ptr;

    // This method of rotation only allows us to rotate within half of the
    // polynomial slots. E.g. for n slots, slot 0 can be rotated to at most
    // n/2-1 and n/2 to n-1. This has implications for how batching is done if
    // performing back propagation under encryption.
    int num_rotation_keys = 1 << (shell_ctx->LogN() - 1);
    int two_n = 1 << (shell_ctx->LogN() + 1);
    v_out.keys.resize(num_rotation_keys);

    auto variance = secret_key->Variance();
    auto t = shell_ctx->PlaintextModulus();

    auto generate_keys_in_range = [&](int start, int end) {
      // Skip rotation key at zero, it does not rotate.
      if (start == 0) ++start;

      uint sub_power = base_power;
      for (int i = 1; i < start; ++i) {
        sub_power *= base_power;
        sub_power %= two_n;
      }

      for (int i = start; i < end; ++i) {
        OP_REQUIRES_VALUE(
            RotationKey k, op_ctx,
            RotationKey::CreateForBgv(*secret_key, sub_power, variance,
                                      gadget_ptr.get(), t, kPrngType));
        v_out.keys[i] = std::move(std::make_shared<RotationKey>(k));
        sub_power *= base_power;
        sub_power %= two_n;
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_key = 70031909;  // ns measured on log_n = 11, 3 moduli
    thread_pool->ParallelFor(num_rotation_keys, cost_per_key,
                             generate_keys_in_range);

    out->scalar<Variant>()() = std::move(v_out);
  }
};

template <typename T>
class RollOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit RollOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    OP_REQUIRES_VALUE(RotationKeyVariant<T> const* rotation_key_var, op_ctx,
                      GetVariant<RotationKeyVariant<T>>(op_ctx, 1));

    std::vector<std::shared_ptr<RotationKey>> const& keys =
        rotation_key_var->keys;

    Tensor const& value = op_ctx->input(2);

    OP_REQUIRES_VALUE(int64 shift, op_ctx, GetScalar<int64>(op_ctx, 3));
    shift = -shift;  // tensorflow.roll() uses negative shift for left shift.

    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot roll empty ciphertext."));

    auto flat_value = value.flat<Variant>();

    // Allocate the output tensor which is the same size as the input tensor.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, value.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Recover num_slots from first ciphertext to validate shift argument.
    SymmetricCtVariant<T> const* ct_var =
        std::move(flat_value(0).get<SymmetricCtVariant<T>>());
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();
    int num_components = ct.NumModuli();

    OP_REQUIRES(op_ctx, abs(shift) < num_slots / 2,
                InvalidArgument("Shifting by too many slots, shift of '", shift,
                                "' must be less than '", num_slots / 2, "'"));

    // Handle negative shift.
    // Careful with c++ modulo operator on negative numbers.
    if (shift < 0) {
      shift += num_slots / 2;
    }

    RotationKey const* key;
    if (shift != 0) {
      OP_REQUIRES(op_ctx, shift < static_cast<int64>(keys.size()),
                  InvalidArgument("No key for shift of '", shift, "'"));
      key = keys[shift].get();
    }

    auto roll_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_var =
            std::move(flat_value(i).get<SymmetricCtVariant<T>>());
        OP_REQUIRES(
            op_ctx, ct_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct = ct_var->ct;

        if (shift == 0) {
          SymmetricCtVariant ct_out_var(ct, shell_ctx_var->ct_context_,
                                        shell_ctx_var->error_params_);
          flat_output(i) = std::move(ct_out_var);
        } else {
          OP_REQUIRES_VALUE(auto ct_sub, op_ctx,
                            ct.Substitute(key->SubstitutionPower()));
          OP_REQUIRES_VALUE(auto ct_rot, op_ctx, key->ApplyTo(ct_sub));

          SymmetricCtVariant ct_out_var(std::move(ct_rot),
                                        shell_ctx_var->ct_context_,
                                        shell_ctx_var->error_params_);
          flat_output(i) = std::move(ct_out_var);
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_rot =
        500 * num_slots * num_components;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_rot,
                             roll_in_range);
  }
};

// Performs a reduce sum over the packing dimension of a ciphertext. This
// requires rotating the ciphertexts log_2(n) times, summing after each
// rotation. The rotation is performed using Galois key-switching keys and the
// output ciphertext is valid under the original secret key.
template <typename T>
class ReduceSumByRotationOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit ReduceSumByRotationOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Recover the inputs.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));

    OP_REQUIRES_VALUE(RotationKeyVariant<T> const* rotation_key_var, op_ctx,
                      GetVariant<RotationKeyVariant<T>>(op_ctx, 1));

    std::vector<std::shared_ptr<RotationKey>> const& keys =
        rotation_key_var->keys;
    Tensor const& value = op_ctx->input(2);
    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot reduce_sum an empty ciphertext."));

    auto flat_value = value.flat<Variant>();

    // Recover num_slots from first ciphertext.
    SymmetricCtVariant<T> const* ct_var =
        std::move(flat_value(0).get<SymmetricCtVariant<T>>());
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();
    int num_components = ct.NumModuli();

    // Allocate the output tensor which is the same size as the input tensor,
    // TensorFlow's reduce_sum has slightly different semantics than this
    // operation. This operation affects top and bottom halves independently, as
    // well as repeating the sum across the halves such that the output is
    // the same shape as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, value.shape(), &output));
    auto flat_output = output->flat<Variant>();

    auto reduce_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        // Learn how many slots there are from first ciphertext and create a
        // deep copy to hold the sum.
        SymmetricCtVariant<T> const* ct_var =
            std::move(flat_value(i).get<SymmetricCtVariant<T>>());
        OP_REQUIRES(op_ctx, ct_var != nullptr,
                    InvalidArgument(
                        "SymmetricCtVariant a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt sum = ct_var->ct;  // deep copy to start the sum.

        // Add the rotations to the sum.
        // Note the ciphertext rotations operate on each half of the
        // ciphertext separately. So the max rotation is by half the number
        // of slots.
        for (int shift = 1; shift < num_slots / 2; shift <<= 1) {
          OP_REQUIRES(
              op_ctx,
              shift < static_cast<int64>(keys.size()),
              InvalidArgument("No key for shift of '", shift, "'"));
          auto key = keys[shift];

          // Rotate by the shift.
          OP_REQUIRES_VALUE(auto ct_sub, op_ctx,
                            sum.Substitute(key->SubstitutionPower()));
          OP_REQUIRES_VALUE(auto ct_rot, op_ctx, key->ApplyTo(ct_sub));

          // Add to the sum.
          OP_REQUIRES_OK(op_ctx, sum.AddInPlace(ct_rot));
        }

        SymmetricCtVariant ct_out_var(std::move(sum),
                                      shell_ctx_var->ct_context_,
                                      shell_ctx_var->error_params_);
        flat_output(i) = std::move(ct_out_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_reduce =
        9000 * num_slots * num_components;  // ns measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_reduce,
                             reduce_in_range);
  }
};

// Performs a reduce sum operation on a ciphertext where the axis to reduce over
// is not the ciphertext packing dimension. As such, this operation does not
// require ciphertext rotations, just addition.
template <typename T>
class ReduceSumOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

  int dim_to_reduce;

 public:
  explicit ReduceSumOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    // Get the dimension to reduce over from the op attributes.
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("axis", &dim_to_reduce));

    // Recall first dimension of a shell variant tensor is the packing
    // dimension. We don't allow expanding this dimension.
    OP_REQUIRES(
        op_ctx, dim_to_reduce != 0,
        InvalidArgument("ReduceSumOp cannot reduce over packing axis (zero'th "
                        "dimension). See ReduceSumByRotationOp."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Recover the inputs.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));

    Tensor const& value = op_ctx->input(1);
    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot reduce_sum an empty ciphertext."));

    OP_REQUIRES(
        op_ctx, dim_to_reduce != 0,
        InvalidArgument("ReduceSumOp cannot reduce over packing axis (zero'th "
                        "dimension). See ReduceSumByRotationOp."));

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    int clamped_dim = dim_to_reduce;
    if (clamped_dim < 0) {
      clamped_dim += value.dims() + 1;  // + 1 for packing dim.
    } else if (clamped_dim > 0) {
      clamped_dim -= 1;  // -1 for packing dimension.
    }

    // Check axis is within dim size.
    OP_REQUIRES(
        op_ctx, clamped_dim >= 0 && clamped_dim < value.dims(),
        InvalidArgument("Cannot reduce_sum over polynomial_axis '", clamped_dim,
                        "for input with shape ", value.shape().DebugString()));

    uint8_t dim_sz_to_reduce = value.dim_size(clamped_dim);

    auto flat_value = value.flat_inner_outer_dims<Variant>(clamped_dim - 1);

    // Setup the output.
    Tensor* output;
    auto output_shape = value.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.RemoveDimWithStatus(clamped_dim));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    // Setup a shape to access the output Tensor as a flat Tensor, with the
    // same indexing as the input Tensor excluding the dimension to reduce.
    int inner_shape = flat_value.dimension(0);
    int outer_shape = flat_value.dimension(2);
    auto flat_output = output->shaped<Variant, 2>({inner_shape, outer_shape});

    auto reduce_in_range = [&](int start, int end) {
      int i_start = start / outer_shape;
      int i_end = end / outer_shape;
      int j_start = start % outer_shape;
      int j_end = end % outer_shape;

      int j_end_wrapped = j_end;
      if (i_start != i_end) {  // Handle wrap around.
        j_end = outer_shape;
      }

      // Take the first ciphertext in the chip and add all the other chips to
      // it.
      for (int i = i_start; i <= i_end; ++i) {
        if (i == i_end) {  // Last row needs special end column;
          j_end = j_end_wrapped;
        }

        for (int j = j_start; j < j_end; ++j) {
          // Get the first chip.
          SymmetricCtVariant<T> const* first_ct_var =
              std::move(flat_value(i, 0, j).get<SymmetricCtVariant<T>>());
          OP_REQUIRES(op_ctx, first_ct_var != nullptr,
                      InvalidArgument(
                          "SymmetricCtVariant a did not unwrap successfully."));
          OP_REQUIRES_OK(op_ctx,
                         const_cast<SymmetricCtVariant<T>*>(first_ct_var)
                             ->MaybeLazyDecode(shell_ctx_var->ct_context_,
                                               shell_ctx_var->error_params_));
          SymmetricCt sum = first_ct_var->ct;  // deep copy to start the sum.

          // Add the remaining chips.
          for (int chip_dim = 1; chip_dim < dim_sz_to_reduce; ++chip_dim) {
            SymmetricCtVariant<T> const* ct_var = std::move(
                flat_value(i, chip_dim, j).get<SymmetricCtVariant<T>>());
            OP_REQUIRES(
                op_ctx, ct_var != nullptr,
                InvalidArgument(
                    "SymmetricCtVariant a did not unwrap successfully."));
            OP_REQUIRES_OK(
                op_ctx,
                const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
            SymmetricCt const& ct = ct_var->ct;

            // Perform the addition.
            OP_REQUIRES_OK(op_ctx, sum.AddInPlace(ct));
          }

          // Store in the output.
          SymmetricCtVariant res_var(std::move(sum), shell_ctx_var->ct_context_,
                                     shell_ctx_var->error_params_);
          flat_output(i, j) = std::move(res_var);
        }

        // Reset the starting column for the next row.
        j_start = 0;
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_reduce =
        10000 * dim_sz_to_reduce;  // ns measured on log_n = 11
    thread_pool->ParallelFor(inner_shape * outer_shape, cost_per_reduce,
                             reduce_in_range);
  }
};

REGISTER_KERNEL_BUILDER(Name("RotationKeyGen64").Device(DEVICE_CPU),
                        RotationKeyGenOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("Roll64").Device(DEVICE_CPU), RollOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ReduceSumByRotation64").Device(DEVICE_CPU),
                        ReduceSumByRotationOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ReduceSum64").Device(DEVICE_CPU),
                        ReduceSumOp<uint64>);