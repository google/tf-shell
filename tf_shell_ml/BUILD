load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package")

filegroup(
    name = "tf_shell_ml_files",
    srcs = glob(
        [
            "*.py",
        ],
        exclude = [
            "test",
        ],
    ),
)

py_library(
    name = "tf_shell_ml",
    srcs = [
        ":tf_shell_ml_files",
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
