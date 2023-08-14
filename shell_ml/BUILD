load("@rules_python//python:packaging.bzl", "py_package")

py_library(
    name = "shell_ml",
    srcs = [
        "__init__.py",
        "activation.py",
        "dense.py",
        "loss.py",
        "optimizer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//shell_tensor:shell_tensor_lib",
    ],
)

py_package(
    name = "shell_ml_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":shell_ml",
    ],
)
