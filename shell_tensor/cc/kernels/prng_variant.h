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
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

using tensorflow::VariantTensorData;

class PrngVariant {
  using Prng = rlwe::SecurePrng;
  using HkdfPrng = rlwe::SingleThreadHkdfPrng;

 public:
  PrngVariant() = default;

  PrngVariant(std::unique_ptr<Prng> p) : prng(std::move(prng)){};

  PrngVariant(PrngVariant const& other) {
    // create a new prng when copied ignoring seed
    prng = HkdfPrng::Create(HkdfPrng::GenerateSeed().value()).value();
  };

  PrngVariant& operator=(PrngVariant const& other) {
    // Guard self assignment
    if (this == &other) return *this;

    prng = HkdfPrng::Create(HkdfPrng::GenerateSeed().value()).value();
    return *this;
  }

  // TODO(jchoncholas): rule of five

  static inline char const kTypeName[] = "ShellPrngVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellPrngVariant"; }

  std::unique_ptr<Prng> prng;
};
