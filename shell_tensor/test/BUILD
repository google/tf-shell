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
    srcs = [
        "basic_test.py",
        "test_utils.py",
    ],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "add_test",
    srcs = [
        "add_test.py",
        "test_utils.py",
    ],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mul_test",
    srcs = [
        "mul_test.py",
        "test_utils.py",
    ],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "composite_test",
    srcs = [
        "composite_test.py",
        "test_utils.py",
    ],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "rotation_test",
    srcs = [
        "rotation_test.py",
        "test_utils.py",
    ],
    deps = [
        "//shell_tensor:shell_tensor_lib",
        requirement("tensorflow-cpu"),
    ],
)
