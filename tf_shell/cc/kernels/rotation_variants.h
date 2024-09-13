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

#include "shell_encryption/rns/rns_galois_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "utils_serdes.h"

using tensorflow::VariantTensorData;

template <typename T>
class RotationKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Gadget = rlwe::RnsGadget<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

 public:
  RotationKeyVariant() {}

  // Create with gadget first, then create and add keys.
  RotationKeyVariant(Gadget gadget) : gadget(gadget) {}

  static inline char const kTypeName[] = "ShellRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellRotationKeyVariant"; }

  Gadget gadget;
  std::vector<RotationKey> keys;
};

template <typename T>
class SingleRotationKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

 public:
  SingleRotationKeyVariant() {}

  // Create with gadget first, then create and add keys.
  SingleRotationKeyVariant(RotationKey key) : key(key) {}

  static inline char const kTypeName[] = "SingleRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "SingleRotationKeyVariant"; }

  RotationKey key;
};

template <typename T>
class FastRotationKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;

 public:
  FastRotationKeyVariant() {}

  // Create with gadget first, then create and add keys.
  FastRotationKeyVariant(std::vector<RnsPolynomial> keys,
                         std::vector<PrimeModulus const*> prime_moduli)
      : keys(std::move(keys)), prime_moduli(std::move(prime_moduli)) {}

  static inline char const kTypeName[] = "ShellFastRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    data->tensors_.reserve(keys.size() + 3);

    SerializePrimeModuli<T>(data, prime_moduli, keys[0].LogN());

    for (auto const& key : keys) {
      auto serialized_key_or = key.Serialize(prime_moduli);
      if (!serialized_key_or.ok()) {
        std::cout << "ERROR: Failed to serialize key: "
                  << serialized_key_or.status();
        return;
      }
      std::string serialized_key;
      serialized_key_or.value().SerializeToString(&serialized_key);
      data->tensors_.push_back(Tensor(serialized_key));
    }
  };

  bool Decode(VariantTensorData const& data) {
    size_t num_keys = data.tensors_.size() - 3;
    keys.reserve(num_keys);

    // Recover the prime moduli.
    prime_moduli.clear();
    int log_n;
    if (!DeerializePrimeModuli<T>(data, 0, prime_moduli, log_n)) {
      return false;
    }

    for (size_t i = 3; i < data.tensors_.size(); ++i) {
      std::string const serialized_key(
          data.tensors_[i].scalar<tstring>()().begin(),
          data.tensors_[i].scalar<tstring>()().end());
      rlwe::SerializedRnsPolynomial serialized_key_polynomial;
      bool ok = serialized_key_polynomial.ParseFromString(serialized_key);
      // std::cout << serialized_key_polynomial.DebugString() << std::endl;
      if (!ok) {
        std::cout << "ERROR: Failed to parse key polynomial." << std::endl;
        return false;
      }
      auto key_polynomial_or = rlwe::RnsPolynomial<ModularInt>::Deserialize(
          serialized_key_polynomial, prime_moduli);
      if (!key_polynomial_or.ok()) {
        std::cout << "ERROR: Failed to deserialize key polynomial: "
                  << key_polynomial_or.status();
        return false;
      }
      keys.push_back(std::move(key_polynomial_or.value()));
    }

    return true;
  };

  std::string DebugString() const { return "ShellFastRotationKeyVariant"; }

  std::vector<RnsPolynomial> keys;
  std::vector<rlwe::PrimeModulus<ModularInt> const*> prime_moduli;
};