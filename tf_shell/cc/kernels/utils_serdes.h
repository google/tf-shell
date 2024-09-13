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

#include <vector>

#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/transcription.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

using tensorflow::VariantTensorData;

template <typename T>
void SerializePrimeModuli(
    VariantTensorData* data,
    absl::Span<rlwe::PrimeModulus<rlwe::MontgomeryInt<T>> const* const> const&
        moduli_span,
    int log_n) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Int = typename ModularInt::Int;
  constexpr static int const IntNumBits = std::numeric_limits<Int>::digits;

  // Store log_n.
  data->tensors_.push_back(Tensor(log_n));

  // Instead of storing the vector<PrimeModulus>, store just the moduli. Skip
  // the NTT parameters and ModularInt parameters as they can be recomputed.
  data->tensors_.push_back(Tensor(moduli_span.size()));

  // Extract the moduli.
  std::vector<Int> moduli;
  moduli.reserve(moduli_span.size());
  for (auto const& modulus : moduli_span) {
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

  return;
};

template <typename T>
bool DeerializePrimeModuli(
    VariantTensorData const& data, int start_index,
    std::vector<rlwe::PrimeModulus<rlwe::MontgomeryInt<T>> const*>&
        prime_moduli,
    int& log_n) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;
  using Int = typename ModularInt::Int;
  constexpr static int const IntNumBits = std::numeric_limits<Int>::digits;

  if (data.tensors_.size() < start_index + 2) {
    std::cout << "ERROR: Not enough tensors to deserialize prime moduli.";
    return false;
  }

  // Recover log_n.
  log_n = data.tensors_[start_index].scalar<int>()(0);

  // Recover the number of moduli.
  int num_moduli = data.tensors_[start_index + 1].scalar<int>()(0);

  // Recover the raw moduli.
  std::string const moduli_str(
      data.tensors_[start_index + 2].scalar<tstring>()().begin(),
      data.tensors_[start_index + 2].scalar<tstring>()().end());
  std::vector<rlwe::Uint8> moduli_v(moduli_str.begin(), moduli_str.end());
  auto moduli_or = rlwe::TranscribeBits<rlwe::Uint8, Int>(
      moduli_v, num_moduli * IntNumBits, 8, IntNumBits);
  if (!moduli_or.ok()) {
    std::cout << "ERROR: Failed to transcribe bits: " << moduli_or.status();
    return false;
  }

  // Recreate the prime moduli, i.e. a vector of PrimeModulus.
  // std::vector<PrimeModulus const*> prime_moduli;
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

    prime_moduli.push_back(new PrimeModulus{std::move(mod_params_or.value()),
                                            std::move(ntt_params_ptr)});
  }
  return true;
};
