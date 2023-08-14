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
#include "shell_encryption/polynomial.h"

template <typename T>
void declare_primitives(pybind11::module& m, std::string const& typestr) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using NttParams = rlwe::NttParameters<ModularInt>;
  namespace py = pybind11;

  std::string pyclass_mon_name = std::string("MontgomeryInt") + typestr;
  py::class_<ModularInt>(m, pyclass_mon_name.c_str(), py::buffer_protocol())
      .def(py::init<T>())
      .def_static("ImportInt", &ModularInt::ImportInt);

  std::string pyclass_vmon_name = std::string("VectorMontgomeryInt") + typestr;
  py::class_<std::vector<ModularInt>>(m, pyclass_vmon_name.c_str(),
                                      py::buffer_protocol())
      .def("ImportVect",
           [](std::vector<T>& in, ModularIntParams const* p)
               -> rlwe::StatusOr<std::vector<ModularInt>> {
             std::vector<ModularInt> mont(in.size(), ModularInt::ImportZero(p));
             for (size_t i = 0; i < in.size(); ++i) {
               RLWE_ASSIGN_OR_RETURN(mont[i], ModularInt::ImportInt(in[i], p));
             }
             return mont;
           })
      .def("size", &std::vector<ModularInt>::size)
      .def("__len__", &std::vector<ModularInt>::size)
      .def("__setitem__", [](std::vector<ModularInt>& self, unsigned index,
                             ModularInt val) { self[index] = std::move(val); })
      .def("__getitem__", [](std::vector<ModularInt>& self, unsigned index) {
        return self[index];
      });

  std::string pyclass_vi_name = std::string("VectorInt") + typestr;
  py::class_<std::vector<T>>(m, pyclass_vi_name.c_str(), py::buffer_protocol())
      .def(py::init<int>())
      .def("size", &std::vector<T>::size)
      .def("__len__", &std::vector<T>::size)
      .def("__setitem__", [](std::vector<T>& self, unsigned index,
                             T val) { self[index] = std::move(val); })
      .def("__getitem__",
           [](std::vector<T>& self, unsigned index) { return self[index]; });

  std::string pyclass_mp_name = std::string("MontgomeryIntParams") + typestr;
  py::class_<ModularIntParams>(m, pyclass_mp_name.c_str(),
                               py::buffer_protocol())
      .def("Create", &ModularIntParams::Create)
      .def_readonly("modulus", &ModularIntParams::modulus);

  std::string pyclass_poly_name = std::string("Polynomial") + typestr;
  py::class_<Polynomial>(m, pyclass_poly_name.c_str(), py::buffer_protocol())
      .def(py::init<std::vector<ModularInt>>())
      .def(py::init<int, ModularIntParams const*>())
      .def_static(
          "ConvertToNtt",
          py::overload_cast<std::vector<ModularInt>, NttParams const*,
                            ModularIntParams const*>(&Polynomial::ConvertToNtt))
      .def("InverseNtt",
           py::overload_cast<NttParams const*, ModularIntParams const*>(
               &Polynomial::InverseNtt, py::const_));
}
