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
// #include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

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

 public:
  FastRotationKeyVariant() {}

  // Create with gadget first, then create and add keys.
  FastRotationKeyVariant(std::vector<RnsPolynomial> keys) : keys(std::move(keys)) {}

  static inline char const kTypeName[] = "ShellFastRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellFastRotationKeyVariant"; }

  std::vector<RnsPolynomial> keys;
};