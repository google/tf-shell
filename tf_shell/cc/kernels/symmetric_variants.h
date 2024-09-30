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

#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/transcription.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

using tensorflow::VariantTensorData;
using tensorflow::errors::InvalidArgument;

template <typename T>
class SymmetricKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using Int = typename ModularInt::Int;
  constexpr static int const IntNumBits = std::numeric_limits<Int>::digits;

 public:
  SymmetricKeyVariant() {}

  SymmetricKeyVariant(Key&& k, std::shared_ptr<Context const> ct_context_)
      : key(std::make_shared<Key>(std::move(k))), ct_context(ct_context_) {}

  static inline char const kTypeName[] = "ShellSymmetricKeyVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    auto serialized_key_or = key->Key().Serialize(key->Moduli());
    if (!serialized_key_or.ok()) {
      std::cout << "ERROR: Failed to serialize key: "
                << serialized_key_or.status();
      return;
    }
    std::string serialized_key;
    serialized_key_or.value().SerializeToString(&serialized_key);
    data->tensors_.push_back(Tensor(serialized_key));
  };

  bool Decode(VariantTensorData const& data) {
    static bool warning_printed = false;
    if (!warning_printed) {
      std::cout << "WARNING: Deserializing secret key. This should only happen "
                   "on the appropriate party."
                << std::endl;
      warning_printed = true;
    }
    if (data.tensors_.size() != 1) {
      std::cout << "ERROR: Decode SymmetricKeyVariant expected 1 tensor, got "
                << data.tensors_.size() << std::endl;
      return false;
    }

    if (!key_str.empty()) {
      std::cout << "ERROR: Key already decoded." << std::endl;
      return false;
    }

    // Recover the key polynomial.
    key_str = std::string(data.tensors_[0].scalar<tstring>()().begin(),
                          data.tensors_[0].scalar<tstring>()().end());

    return true;
  };

  Status MaybeLazyDecode(std::shared_ptr<Context const> ct_context_,
                         int noise_variance) {
    // If this key has already been fully decoded, nothing to do.
    if (key_str.empty()) {
      return OkStatus();
    }

    rlwe::SerializedRnsPolynomial serialized_key;
    bool ok = serialized_key.ParseFromString(key_str);
    if (!ok) {
      return InvalidArgument("Failed to parse key polynomial.");
    }

    // Using the moduli, reconstruct the key polynomial.
    TF_ASSIGN_OR_RETURN(auto key_polynomial,
                        rlwe::RnsPolynomial<ModularInt>::Deserialize(
                            serialized_key, ct_context_->MainPrimeModuli()));

    // Create the key without having access to the constructor.
    struct RawKey {
      rlwe::RnsPolynomial<ModularInt> key;
      std::vector<rlwe::PrimeModulus<ModularInt> const*> moduli;
      int variance;
    };
    RawKey raw_key{std::move(key_polynomial),
                   std::move(ct_context_->MainPrimeModuli()), noise_variance};
    Key* recovered_key = reinterpret_cast<Key*>(&raw_key);  // UB!
    key = std::make_shared<Key>(*recovered_key);

    // Hold a pointer to the context so the moduli this key depends on wont be
    // deleted if the ContextVariant is deleted before this key.
    ct_context = ct_context_;

    // Clear the serialized key string.
    key_str.clear();

    return OkStatus();
  }

  std::string DebugString() const { return "ShellSymmetricKeyVariant"; }

  std::shared_ptr<Key> key;
  std::string key_str;
  std::shared_ptr<Context const> ct_context;
};

template <typename T>
class SymmetricCtVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using ErrorParams = rlwe::RnsErrorParams<ModularInt>;

 public:
  SymmetricCtVariant() : ct({}, {}, 0, 0, nullptr) {}

  SymmetricCtVariant(SymmetricCt ct, std::shared_ptr<Context const> ct_context_,
                     std::shared_ptr<ErrorParams const> error_params_)
      : ct(std::move(ct)),
        ct_context(ct_context_),
        error_params(error_params_) {}

  static inline char const kTypeName[] = "ShellSymmetricCtVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    // Store the ciphertext.
    auto serialized_ct_or = ct.Serialize();
    if (!serialized_ct_or.ok()) {
      std::cout << "ERROR: Failed to serialize ciphertext: "
                << serialized_ct_or.status() << std::endl;
      return;
    }
    std::string serialized_ct;
    serialized_ct_or.value().SerializeToString(&serialized_ct);
    data->tensors_.push_back(Tensor(serialized_ct));
  };

  // Decoding requires access to a shell RnsContext object. TensorFlow requires
  // that a variant (i.e. this class) must be able to decode itself without
  // external context. This causes an issue because storing the context
  // for every ciphertext is wasteful. To work around this, partially decode
  // the ciphertext polynomials to a string, and wait to complete the decode
  // until an operation is attempted and the context is available.
  bool Decode(VariantTensorData const& data) {
    if (data.tensors_.size() != 1) {
      std::cout << "ERROR: Decode SymmetricCtVariant expected 1 tensor, got "
                << data.tensors_.size() << std::endl;
      return false;
    }

    if (!ct_str.empty()) {
      std::cout << "ERROR: Ciphertext already decoded." << std::endl;
      return false;
    }

    // Recover the serialized ciphertext string.
    ct_str = std::string(data.tensors_[0].scalar<tstring>()().begin(),
                         data.tensors_[0].scalar<tstring>()().end());

    return true;
  };

  Status MaybeLazyDecode(std::shared_ptr<Context const> ct_context_,
                         std::shared_ptr<ErrorParams const> error_params_) {
    // If this ciphertext has already been fully decoded, nothing to do.
    if (ct_str.empty()) {
      return OkStatus();
    }

    rlwe::SerializedRnsRlweCiphertext serialized_ct;
    bool ok = serialized_ct.ParseFromString(ct_str);
    if (!ok) {
      return InvalidArgument("Failed to parse ciphertext.");
    }
    TF_ASSIGN_OR_RETURN(
        auto generic_ct,
        SymmetricCt::Deserialize(serialized_ct, ct_context_->MainPrimeModuli(),
                                 error_params_.get()));
    ct = std::move(static_cast<SymmetricCt>(generic_ct));

    // Hold a pointer to the context and error params so the moduli this
    // ciphertext depends on wont be deleted if the ContextVariant is delected
    // before this ciphertext.
    ct_context = ct_context_;
    error_params = error_params_;

    // Clear the serialized ciphertext string.
    ct_str.clear();

    return OkStatus();
  };

  std::string DebugString() const { return "ShellSymmetricCtVariant"; }

  SymmetricCt ct;
  std::string ct_str;
  std::shared_ptr<Context const> ct_context;
  std::shared_ptr<ErrorParams const> error_params;
};
