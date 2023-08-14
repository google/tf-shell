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

#include <pybind11/pybind11.h>

#include <memory>

#include "binder_context.h"
#include "binder_primitives.h"
#include "binder_prng.h"
#include "binder_symmetric.h"
#include "pybind11_abseil/status_casters.h"
#include "shell_encryption/constants.h"

// Prevent copies of MontgomeryInts between python and c++
PYBIND11_MAKE_OPAQUE(std::vector<rlwe::MontgomeryInt<uint64_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<rlwe::MontgomeryInt<absl::uint128>>);

PYBIND11_MODULE(shell, m) {
  pybind11::google::ImportStatusModule();
  m.doc() = "SHELL Encryption Library Bindings";

  m.attr("kModulus59") = rlwe::kModulus59;
  m.attr("kInvModulus59") = rlwe::kInvModulus59;
  m.attr("kLogDegreeBound59") = rlwe::kLogDegreeBound59;
  m.attr("kDegreeBound59") = rlwe::kDegreeBound59;

  // Cannot create bindings for templated code.
  // Instead, encode the type in the object name, e.g.
  // Context64, ContextParams64
  declare_context<uint64_t>(m, "64");
  // Context128, ContextParams128
  declare_context<absl::uint128>(m, "128");

  // NttParams64, ErrorParams64
  declare_params<uint64_t>(m, "64");
  // NttParams128, ErrorParams128
  declare_params<absl::uint128>(m, "128");

  // SingleThreadHkdfPrng
  declare_prng(m);

  // MontgomeryInt64, VectorMontgomeryInt64, VectorInt64,
  // MontgomeryIntParams64, Polynomial64
  declare_primitives<uint64_t>(m, "64");

  // SymmetricKey64, SymmetricCt64
  declare_symmetric<uint64_t>(m, "64");
  // SymmetricKey128, SymmetricCt128
  declare_symmetric<absl::uint128>(m, "128");
}
