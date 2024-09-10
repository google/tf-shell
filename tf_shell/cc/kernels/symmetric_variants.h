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

template <typename T>
class SymmetricKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using Int = typename ModularInt::Int;
  constexpr static int const IntNumBits = std::numeric_limits<Int>::digits;

 public:
  SymmetricKeyVariant() {}

  SymmetricKeyVariant(Key&& k) { key = std::make_shared<Key>(k); }

  Status Initialize(Key k) {
    key = std::make_shared<Key>(k);
    return OkStatus();
  }

  static inline char const kTypeName[] = "ShellSymmetricKeyVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    data->tensors_.reserve(4);

    // First store the key.
    auto serialized_key_or = key->Key().Serialize(key->Moduli());
    if (!serialized_key_or.ok()) {
      std::cout << "ERROR: Failed to serialize key: "
                << serialized_key_or.status();
      return;
    }
    std::string serialized_key;
    serialized_key_or.value().SerializeToString(&serialized_key);
    data->tensors_.push_back(Tensor(serialized_key));

    // Instead of storing the vector<PrimeModulus>, store just the moduli. Skip
    // the NTT parameters and ModularInt parameters as they can be recomputed.
    data->tensors_.push_back(Tensor(key->Moduli().size()));

    // Extract the moduli.
    std::vector<Int> moduli;
    moduli.reserve(key->Moduli().size());
    for (auto const& modulus : key->Moduli()) {
      moduli.push_back(modulus->Modulus());
    }

    // Encode the moduli as a string.
    auto moduli_bytes_or = rlwe::TranscribeBits<Int, rlwe::Uint8>(
        moduli, moduli.size() * IntNumBits, IntNumBits, 8);
    if (!moduli_bytes_or.ok()) {
      std::cout << "ERROR: Failed to transcribe bits: "
                << moduli_bytes_or.status();
      return;
    }
    std::string moduli_str(
        std::make_move_iterator(moduli_bytes_or.value().begin()),
        std::make_move_iterator(moduli_bytes_or.value().end()));
    data->tensors_.push_back(Tensor(moduli_str));

    // Store the variance of the key.
    data->tensors_.push_back(Tensor(key->Variance()));
  };

  bool Decode(VariantTensorData const& data) {
    if (data.tensors_.size() != 4) {
      return false;
    }

    // Recover the key polynomial.
    std::string const serialized_key(
        data.tensors_[0].scalar<tstring>()().begin(),
        data.tensors_[0].scalar<tstring>()().end());
    rlwe::SerializedRnsPolynomial serialized_key_polynomial;
    bool ok = serialized_key_polynomial.ParseFromString(serialized_key);
    // std::cout << serialized_key_polynomial.DebugString() << std::endl;
    if (!ok) {
      std::cout << "ERROR: Failed to parse key polynomial." << std::endl;
      return false;
    }

    // Recover log_n.
    int log_n = serialized_key_polynomial.log_n();

    // Recover the number of moduli.
    int num_moduli = data.tensors_[1].scalar<int>()(0);

    // Recover the raw moduli.
    std::string const moduli_str(data.tensors_[2].scalar<tstring>()().begin(),
                                 data.tensors_[2].scalar<tstring>()().end());
    std::vector<rlwe::Uint8> moduli_v(moduli_str.begin(), moduli_str.end());
    auto moduli_or = rlwe::TranscribeBits<rlwe::Uint8, Int>(
        moduli_v, num_moduli * IntNumBits, 8, IntNumBits);
    if (!moduli_or.ok()) {
      std::cout << "ERROR: Failed to transcribe bits: " << moduli_or.status();
      return false;
    }

    // Recreate the prime moduli, i.e. a vector of PrimeModulus.
    std::vector<rlwe::PrimeModulus<ModularInt> const*> prime_moduli;
    prime_moduli.reserve(moduli_or.value().size());
    for (auto const& modulus : moduli_or.value()) {
      auto mod_params_or = ModularInt::Params::Create(modulus);
      if (!mod_params_or.ok()) {
        std::cout << "ERROR: Failed to create mod params: "
                  << mod_params_or.status();
        return false;
      }
      auto ntt_params_or = rlwe::InitializeNttParameters<ModularInt>(
          log_n, mod_params_or.value().get());
      if (!ntt_params_or.ok()) {
        std::cout << "ERROR: Failed to initialize NTT parameters: "
                  << ntt_params_or.status();
        return false;
      }
      auto ntt_params_ptr = std::make_unique<rlwe::NttParameters<ModularInt>>(
          std::move(ntt_params_or.value()));

      prime_moduli.push_back(new rlwe::PrimeModulus<ModularInt>{
          std::move(mod_params_or.value()), std::move(ntt_params_ptr)});
    }

    // Using the moduli, reconstruct the key polynomial.
    auto key_polynomial_or = rlwe::RnsPolynomial<ModularInt>::Deserialize(
        serialized_key_polynomial, prime_moduli);
    if (!key_polynomial_or.ok()) {
      std::cout << "ERROR: Failed to deserialize key polynomial: "
                << key_polynomial_or.status();
      return false;
    }

    // Recover the variance.
    int variance = data.tensors_[3].scalar<int>()(0);

    // Create the key without having access to the constructor.
    struct RawKey {
      rlwe::RnsPolynomial<ModularInt> key;
      std::vector<rlwe::PrimeModulus<ModularInt> const*> moduli;
      int variance;
    };
    RawKey raw_key{std::move(key_polynomial_or.value()),
                   std::move(prime_moduli), variance};
    Key* recovered_key = reinterpret_cast<Key*>(&raw_key);  // UB!
    key = std::make_shared<Key>(*recovered_key);

    return true;
  };

  std::string DebugString() const { return "ShellSymmetricKeyVariant"; }

  std::shared_ptr<Key> key;
};

template <typename T>
class SymmetricCtVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  SymmetricCtVariant() {}

  SymmetricCtVariant(SymmetricCt arg) : ct(std::move(arg)) {}

  static inline char const kTypeName[] = "ShellSymmetricCtVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellSymmetricCtVariant"; }

  SymmetricCt ct;
};
