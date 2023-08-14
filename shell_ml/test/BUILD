load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "mnist_plaintext_post_scale_test",
    size = "enormous",
    srcs = ["mnist_plaintext_post_scale_test.py"],
    deps = [
        "//shell_ml",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mnist_shell_backprop_test",
    size = "enormous",
    srcs = ["mnist_shell_backprop_test.py"],
    deps = [
        "//shell_ml",
        requirement("tensorflow-cpu"),
    ],
)
