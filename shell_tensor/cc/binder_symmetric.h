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
#include <pybind11/pybind11.h>

#include <memory>

#include "shell_encryption/montgomery.h"
#include "shell_encryption/symmetric_encryption.h"

template <typename T>
void declare_symmetric(pybind11::module& m, std::string const& typestr) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using KeyClass = rlwe::SymmetricRlweKey<ModularInt>;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using CtClass = rlwe::SymmetricRlweCiphertext<ModularInt>;
  using Prng = rlwe::SingleThreadHkdfPrng;
  namespace py = pybind11;

  std::string pyclass_key_name = std::string("SymmetricKey") + typestr;
  py::class_<KeyClass>(m, pyclass_key_name.c_str(), py::buffer_protocol())
      .def("Sample", [](Context* c, Prng* p) {
        return KeyClass::Sample(c->GetLogN(), c->GetVariance(), c->GetLogT(),
                                c->GetModulusParams(), c->GetNttParams(), p);
      });

  std::string pyclass_ct_name = std::string("SymmetricCt") + typestr;
  py::class_<CtClass>(m, pyclass_ct_name.c_str(), py::buffer_protocol())
      .def("Encrypt",
           [](KeyClass const& key, Polynomial const& polynomial,
              Context const* context, rlwe::SecurePrng* prng) {
             return rlwe::Encrypt<ModularInt>(key, polynomial,
                                              context->GetErrorParams(), prng);
           })
      .def("Encrypt",
           [](KeyClass const& key, std::vector<T> const& values,
              Context const* context,
              rlwe::SecurePrng* prng) -> absl::StatusOr<CtClass> {
             std::vector<ModularInt> mont(
                 values.size(),
                 ModularInt::ImportZero(context->GetModulusParams()));
             for (size_t i = 0; i < mont.size(); ++i) {
               RLWE_ASSIGN_OR_RETURN(
                   mont[i], ModularInt::ImportInt(values[i],
                                                  context->GetModulusParams()));
             }

             auto plaintext_ntt = rlwe::Polynomial<ModularInt>::ConvertToNtt(
                 mont, context->GetNttParams(), context->GetModulusParams());
             return rlwe::Encrypt<ModularInt>(key, plaintext_ntt,
                                              context->GetErrorParams(), prng);
           })
      .def("Decrypt", py::overload_cast<KeyClass const&, CtClass const&>(
                          &rlwe::Decrypt<ModularInt>));
}
