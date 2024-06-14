load("@rules_python//python:packaging.bzl", "py_package")

filegroup(
    name = "shell_ops_src",
    srcs = glob([
        "cc/kernels/*.cc",
        "cc/kernels/*.h",
        "cc/ops/*.cc",
        "cc/ops/*.h",
    ]),
)

cc_binary(
    name = "python/ops/_shell_ops.so",
    srcs = [":shell_ops_src"],
    copts = [
        "-pthread",
        "-fPIC",
    ],
    linkshared = 1,
    deps = [
        "//third_party/tensorflow:hermetic_tf",
        "@shell_encryption//shell_encryption:context",
        "@shell_encryption//shell_encryption:montgomery",
        "@shell_encryption//shell_encryption:polynomial",
        "@shell_encryption//shell_encryption/prng:hkdf_prng",
        "@shell_encryption//shell_encryption/rns:finite_field_encoder",
        "@shell_encryption//shell_encryption/rns:rns_ciphertext",
        "@shell_encryption//shell_encryption/rns:rns_context",
        "@shell_encryption//shell_encryption/rns:rns_error_params",
        "@shell_encryption//shell_encryption/rns:rns_gadget",
        "@shell_encryption//shell_encryption/rns:rns_galois_key",
        "@shell_encryption//shell_encryption/rns:rns_integer",
        "@shell_encryption//shell_encryption/rns:rns_modulus",
        "@shell_encryption//shell_encryption/rns:rns_polynomial",
        "@shell_encryption//shell_encryption/rns:rns_secret_key",
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

cc_binary(
    name = "python/optimizers/_ct_pt_optimizer.so",
    srcs = [
        "cc/optimizers/ct_pt.cc",
        "cc/optimizers/ct_pt.h",
    ],
    copts = [
        "-pthread",
        "-fPIC",
    ],
    linkshared = 1,
    deps = [
        "//third_party/tensorflow:hermetic_tf",
        "@com_google_protobuf//:protobuf",
    ],
)

py_library(
    name = "shell_optimizers_py",
    srcs = [
        "python/optimizers/shell_optimizers.py",
    ],
    data = [
        ":python/optimizers/_ct_pt_optimizer.so",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "tf_shell_lib",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
        "python/optimizers/__init__.py",
        "python/shell_context.py",
        "python/shell_key.py",
        "python/shell_tensor.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":shell_ops_py",
        ":shell_optimizers_py",
    ],
)

py_package(
    name = "tf_shell_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":tf_shell_lib",
    ],
)
