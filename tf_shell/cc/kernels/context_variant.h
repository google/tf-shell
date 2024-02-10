/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>

#include "absl/status/status.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/coefficient_encoder.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_error_params.h"
#include "shell_encryption/rns/rns_gadget.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/rns/rns_secret_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "utils.h"

using tensorflow::VariantTensorData;

// This class wraps SHELL encryption library objects to store state required for
// performing homomorphic operations on encrypted data.
//
// Tensorflow requires copy constructors be defined while many of objects in
// this class from the SHELL encryption library have their copy constructors
// deleted to prevent users from performance issues by accidental copies.
// As such, the copy constructors defined here need to create copies of the
// shell objects without calling the copy constructors.
template <typename T>
class ContextVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  // static_assert(std::is_same<T, typename ModularInt::Int>::value);
  using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using ErrorParams = rlwe::RnsErrorParams<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Gadget = rlwe::RnsGadget<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using HkdfPrng = rlwe::SingleThreadHkdfPrng;

 public:
  ContextVariant() = default;

  ~ContextVariant() = default;

  ContextVariant(ContextVariant const& other) : ContextVariant() {
    // TODO(jchoncholas): Consider seeding new prng with value from current prng
    // on copy to preserve seed-like behavior for initially seeded prngs.
    std::string seed{};
    Initialize(other.log_n_, other.qs_, other.ps_, other.pt_modulus_,
               other.noise_variance_, seed)
        .ok();
  }

  ContextVariant& operator=(ContextVariant const& other) {
    // if (*this == other) return *this;  // guard self-assignment

    // TODO(jchoncholas): Consider seeding new prng with value from current prng
    // on copy to preserve seed-like behavior for initially seeded prngs.
    std::string seed{};
    Initialize(other.log_n_, other.qs_, other.ps_, other.pt_modulus_,
               other.noise_variance_, seed)
        .ok();

    return *this;
  }

  absl::Status Initialize(size_t log_n, std::vector<T> qs, std::vector<T> ps,
                          T pt_modulus, size_t noise_variance,
                          std::string& seed = {}) {
    // Store necessary information for copy constructor.
    log_n_ = log_n;
    qs_ = qs;
    ps_ = ps;
    pt_modulus_ = pt_modulus;
    noise_variance_ = noise_variance;

    // Create plaintext context objects
    TF_SHELL_ASSIGN_OR_RETURN(pt_params_, ModularIntParams::Create(pt_modulus));
    TF_SHELL_ASSIGN_OR_RETURN(
        pt_ntt_params_,
        rlwe::InitializeNttParameters<ModularInt>(log_n, pt_params_.get()));

    // Create ciphertext context objects
    TF_SHELL_ASSIGN_OR_RETURN(
        auto ct_context,
        Context::CreateForBgvFiniteFieldEncoding(log_n, qs, ps, pt_modulus));
    ct_context_ = std::make_unique<Context const>(std::move(ct_context));

    int log_t = floor(std::log2(static_cast<double>(pt_modulus)));
    TF_SHELL_ASSIGN_OR_RETURN(
        auto error_params,
        ErrorParams::Create(
            log_n, ct_context_->MainPrimeModuli(),
            /*aux_moduli=*/{},  // TODO(jchoncholas): should this be ps?
            log_t, sqrt(noise_variance)));
    error_params_ =
        std::make_unique<ErrorParams const>(std::move(error_params));

    TF_SHELL_ASSIGN_OR_RETURN(auto encoder, Encoder::Create(ct_context_.get()));
    encoder_ = std::make_unique<Encoder const>(std::move(encoder));

    int level = qs.size() - 1;
    TF_SHELL_ASSIGN_OR_RETURN(auto q_hats,
                              ct_context_->MainPrimeModulusComplements(level));
    TF_SHELL_ASSIGN_OR_RETURN(auto q_hat_invs,
                              ct_context_->MainPrimeModulusCrtFactors(level));
    std::vector<size_t> log_bs(qs.size(), ContextVariant<T>::kLogGadgetBase);
    TF_SHELL_ASSIGN_OR_RETURN(auto gadget,
                              Gadget::Create(log_n, log_bs, q_hats, q_hat_invs,
                                             ct_context_->MainPrimeModuli()));
    gadget_ = std::make_unique<Gadget const>(std::move(gadget));

    if (seed.empty()) {
      TF_SHELL_ASSIGN_OR_RETURN(auto gen_seed, HkdfPrng::GenerateSeed());
      TF_SHELL_ASSIGN_OR_RETURN(prng_, HkdfPrng::Create(gen_seed));
    } else {
      TF_SHELL_ASSIGN_OR_RETURN(prng_, HkdfPrng::Create(seed));
    }

    return absl::OkStatus();
  }

  static inline char const kTypeName[] = "ShellContextVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellContextVariant"; }

  static constexpr int kLogGadgetBase = 10;

  size_t log_n_;
  std::vector<T> qs_;
  std::vector<T> ps_;
  T pt_modulus_;
  size_t noise_variance_;

  // Ideally these members wouldn't be smart pointers (plain pointers or even
  // just the objects), but many of them don't have default constructors and
  // in some places SHELL expects callers to use smart pointers.
  std::unique_ptr<ModularIntParams const> pt_params_;
  NttParams pt_ntt_params_;
  std::unique_ptr<Context const> ct_context_;
  std::unique_ptr<ErrorParams const> error_params_;
  std::unique_ptr<Encoder const> encoder_;
  std::unique_ptr<Gadget const> gadget_;
  std::unique_ptr<Prng> prng_;
};
