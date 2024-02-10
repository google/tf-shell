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

#include "shell_encryption/context.h"
#include "shell_encryption/error_params.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/ntt_parameters.h"

template <typename T>
void declare_context(pybind11::module &m, std::string const &typestr) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RlweContext<ModularInt>;
  using ContextParams = struct Context::Parameters;
  namespace py = pybind11;

  std::string pyclass_name = std::string("Context") + typestr;
  py::class_<Context> boundContext(m, pyclass_name.c_str(),
                                   py::buffer_protocol());
  boundContext.def("Create", &Context::Create)
      .def("GetModulus", &Context::GetModulus)
      .def("GetLogT", &Context::GetLogT)
      .def("GetLogN", &Context::GetLogN)
      .def("GetVariance", &Context::GetVariance)
      .def("GetModulusParams", &Context::GetModulusParams,
           py::return_value_policy::reference)
      .def("GetNttParams", &Context::GetNttParams,
           py::return_value_policy::reference)
      .def("GetErrorParams", &Context::GetErrorParams,
           py::return_value_policy::reference);

  std::string pyclass_p_name = std::string("ContextParams") + typestr;
  py::class_<ContextParams>(m, pyclass_p_name.c_str(), boundContext,
                            py::buffer_protocol())
      .def(py::init<>(
               [](T modulus, size_t log_n, size_t log_t, size_t variance) {
                 return ContextParams{modulus, log_n, log_t, variance};
               }),
           py::arg("modulus"), py::arg("log_n"), py::arg("log_t"),
           py::arg("variance"))
      .def_readwrite("modulus", &ContextParams::modulus)
      .def_readwrite("log_n", &ContextParams::log_n)
      .def_readwrite("log_t", &ContextParams::log_t)
      .def_readwrite("variance", &ContextParams::variance);
}

template <typename T>
void declare_params(pybind11::module &m, std::string const &typestr) {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using ErrorParams = rlwe::ErrorParams<ModularInt>;
  namespace py = pybind11;

  std::string pyclass_ntt_name = std::string("NttParams") + typestr;
  py::class_<NttParams>(m, pyclass_ntt_name.c_str(), py::buffer_protocol())
      .def("Initialize", &rlwe::InitializeNttParameters<ModularInt>,
           py::return_value_policy::move)
      .def_readwrite("number_coeffs", &NttParams::number_coeffs);

  std::string pyclass_err_name = std::string("ErrorParams") + typestr;
  py::class_<ErrorParams>(m, pyclass_err_name.c_str(), py::buffer_protocol())
      .def("Create", &ErrorParams::Create, py::return_value_policy::move)
      .def("B_plaintext",
           py::overload_cast<>(&ErrorParams::B_plaintext, py::const_))
      .def("B_encryption",
           py::overload_cast<>(&ErrorParams::B_encryption, py::const_))
      .def("B_scale", py::overload_cast<>(&ErrorParams::B_scale, py::const_));
}
