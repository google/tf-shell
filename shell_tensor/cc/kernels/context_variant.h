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
#include "shell_encryption/context.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

using tensorflow::VariantTensorData;

template <typename T>
class ContextVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using Context = rlwe::RlweContext<ModularInt>;

 public:
  ContextVariant() = default;

  ContextVariant(ContextVariant const& other) {
    pt_params =
        std::move(ModularIntParams::Create(other.pt_params->modulus).value());

    pt_ntt_params = rlwe::InitializeNttParameters<ModularInt>(
                        other.ct_context->GetLogN(), pt_params.get())
                        .value();

    typename Context::Parameters c{
        other.ct_context->GetModulus(), other.ct_context->GetLogN(),
        other.ct_context->GetLogT(), other.ct_context->GetVariance()};
    ct_context = Context::Create(c).value();  // deep copy
  }

  ContextVariant& operator=(ContextVariant const& other) {
    // Guard self assignment
    if (this == &other) return *this;

    pt_params =
        std::move(ModularIntParams::Create(other.pt_params->modulus).value());

    pt_ntt_params = rlwe::InitializeNttParameters<ModularInt>(
                        other.ct_context->GetLogN(), pt_params.get())
                        .value();

    typename Context::Parameters c{
        other.ct_context->GetModulus(), other.ct_context->GetLogN(),
        other.ct_context->GetLogT(), other.ct_context->GetVariance()};
    ct_context = Context::Create(c).value();  // deep copy

    return *this;
  }

  // TODO(jchoncholas): rule of five

  static inline char const kTypeName[] = "ShellContextVariant";

  std::string TypeName() const { return kTypeName; }

  // TODO(jchoncholas): implement for networking
  void Encode(VariantTensorData* data) const {};

  // TODO(jchoncholas): implement for networking
  bool Decode(VariantTensorData const& data) { return true; };

  std::string DebugString() const { return "ShellContextVariant"; }

  std::unique_ptr<const ModularIntParams> pt_params;
  NttParams pt_ntt_params;
  std::unique_ptr<const Context> ct_context;
};
