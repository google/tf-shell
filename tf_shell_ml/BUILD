load("@rules_python//python:packaging.bzl", "py_package")

py_library(
    name = "tf_shell_ml",
    srcs = [
        "__init__.py",
        "activation.py",
        "dense.py",
        "dropout.py",
        "embedding.py",
        "globalaveragepool1d.py",
        "loss.py",
        "model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tf_shell:tf_shell_lib",
    ],
)

py_package(
    name = "tf_shell_ml_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":tf_shell_ml",
    ],
)
