load("@rules_python//python:packaging.bzl", "py_package")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_binary(
    name = "python/ops/_shell_ops.so",
    srcs = [
        "cc/kernels/add_kernels.cc",
        "cc/kernels/context_import_kernel.cc",
        "cc/kernels/context_variant.h",
        "cc/kernels/mul_kernels.cc",
        "cc/kernels/polynomial_kernels.cc",
        "cc/kernels/polynomial_variant.h",
        "cc/kernels/prng_kernel.cc",
        "cc/kernels/prng_variant.h",
        "cc/kernels/symmetric_kernels.cc",
        "cc/kernels/symmetric_variants.h",
        "cc/kernels/utils.h",
        "cc/ops/shell_ops.cc",
    ],
    copts = [
        "-pthread",
        "-fPIC",
    ],
    linkshared = 1,
    deps = [
        "//third_party/tensorflow:hermetic_tf",
        "@shell_encryption//shell_encryption:context",
        "@shell_encryption//shell_encryption:modulus_conversion",
        "@shell_encryption//shell_encryption:montgomery",
        "@shell_encryption//shell_encryption:polynomial",
        "@shell_encryption//shell_encryption:symmetric_encryption",
        "@shell_encryption//shell_encryption/prng:single_thread_hkdf_prng",
    ],
)

py_library(
    name = "shell_ops_py",
    srcs = [
        "python/ops/shell_ops.py",
    ],
    data = [
        ":python/ops/_shell_ops.so",
    ],
    srcs_version = "PY3",
)

pybind_extension(
    name = "shell",
    srcs = [
        "cc/binder.cc",
        "cc/binder_context.h",
        "cc/binder_primitives.h",
        "cc/binder_prng.h",
        "cc/binder_symmetric.h",
    ],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@shell_encryption//shell_encryption:context",
        "@shell_encryption//shell_encryption:montgomery",
        "@shell_encryption//shell_encryption:polynomial",
        "@shell_encryption//shell_encryption:statusor_fork",
        "@shell_encryption//shell_encryption:symmetric_encryption",
        "@shell_encryption//shell_encryption/prng:single_thread_hkdf_prng",
    ],
)

py_library(
    name = "shell_tensor_lib",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/shell_tensor.py",
    ],
    data = [
        ":shell.so",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":shell_ops_py",
    ],
)

py_package(
    name = "shell_tensor_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":shell_tensor_lib",
    ],
)
