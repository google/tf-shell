load("@pip//:requirements.bzl", "requirement")
load("//tools:dynamic_genrule.bzl", "dynamic_genrule")

py_binary(
    name = "extract_tf_includes",
    srcs = [":extract_tf_includes.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow-cpu"),
    ],
)

py_binary(
    name = "extract_tf_lib",
    srcs = [":extract_tf_lib.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow-cpu"),
    ],
)

dynamic_genrule(
    name = "hermetic_tf_headers",
    tool = ":extract_tf_includes",
)

genrule(
    name = "hermetic_tf_lib",
    outs = ["libtensorflow_framework.so.2"],
    cmd = "$(location :extract_tf_lib) $@",
    tools = [":extract_tf_lib"],
)

cc_library(
    name = "tf_headers",
    hdrs = [":hermetic_tf_headers"],
    includes = ["hermetic_tf_headers/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_lib",
    data = [":hermetic_tf_lib"],
    visibility = ["//visibility:public"],
    deps = [":hermetic_tf_lib"],
)

cc_import(
    name = "hermetic_tf",
    hdrs = [":tf_headers"],
    shared_library = "libtensorflow_framework.so.2",
    visibility = ["//visibility:public"],
    deps = [
        ":tf_headers",
        ":tf_lib",
    ],
    alwayslink = 1,
)
