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

#include "pybind11_abseil/status_casters.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

void declare_prng(pybind11::module &m) {
  using Prng = rlwe::SecurePrng;
  using HkdfPrng = rlwe::SingleThreadHkdfPrng;
  namespace py = pybind11;

  std::string pyclass_sp_name = std::string("SecurePrng");
  py::class_<Prng>(m, pyclass_sp_name.c_str(), py::buffer_protocol());

  std::string pyclass_name = std::string("SingleThreadHkdfPrng");
  py::class_<HkdfPrng, Prng>(m, pyclass_name.c_str(), py::buffer_protocol())
      .def("Create", &HkdfPrng::Create)
      .def("Rand8", &HkdfPrng::Rand8)
      .def("Rand64", &HkdfPrng::Rand64)
      .def("GenerateSeed",
           []() -> rlwe::StatusOr<py::bytes> {
             RLWE_ASSIGN_OR_RETURN(std::string seed, HkdfPrng::GenerateSeed());
             return py::bytes(
                 seed);  // seed is not utf8 so must be returned as bytes
           })
      .def("SeedLength", &HkdfPrng::SeedLength);
}
