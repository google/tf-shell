load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "context_test",
    size = "small",
    srcs = [
        "context_test.py",
    ],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "basic_test",
    size = "small",
    srcs = [
        "basic_test.py",
        "test_utils.py",
    ],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "add_test",
    size = "medium",
    srcs = [
        "add_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mul_test",
    size = "medium",
    srcs = [
        "mul_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mat_mul_test",
    size = "medium",
    srcs = [
        "mat_mul_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "composite_test",
    size = "small",
    srcs = [
        "composite_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "rotation_test",
    size = "large",
    srcs = [
        "rotation_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "rotation_test_fast",
    size = "large",
    srcs = [
        "rotation_test_fast.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "shape_test",
    size = "small",
    srcs = [
        "shape_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "segment_test",
    size = "enormous",
    srcs = [
        "segment_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "ct_pt_optimizer_test",
    size = "medium",
    srcs = [
        "ct_pt_optimizer_test.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "pt_pt_optimizer_test",
    size = "medium",
    srcs = [
        "pt_pt_optimizer_test.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "auto_param_optimizer_test",
    size = "medium",
    srcs = [
        "auto_param_optimizer_test.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "clip_and_noise_test",
    size = "medium",
    srcs = [
        "clip_and_noise_test.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "distribution_test",
    size = "medium",
    srcs = [
        "distribution_test.py",
    ],
    imports = ["./"],
    deps = [
        "//tf_shell:tf_shell_lib",
        requirement("tensorflow-cpu"),
    ],
)
