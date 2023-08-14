load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "bindings_test",
    srcs = ["bindings_test.py"],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "basic_test",
    srcs = ["basic_test.py"],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "add_test",
    srcs = ["add_test.py"],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mul_test",
    srcs = ["shell_tensor_mul.py"],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)
