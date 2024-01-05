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
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

using tensorflow::VariantTensorData;

template <typename T>
class SymmetricKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;

 public:
  SymmetricKeyVariant() {}

  SymmetricKeyVariant(Key k) : key(k) {}

  static inline char const kTypeName[] = "ShellSymmetricKeyVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellSymmetricKeyVariant"; }

  Key key;
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
