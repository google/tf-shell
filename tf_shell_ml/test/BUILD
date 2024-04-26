load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "mnist_post_scale_test",
    size = "large",
    srcs = ["mnist_post_scale_test.py"],
    deps = [
        "//tf_shell_ml",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mnist_noenc_backprop_test",
    size = "large",
    srcs = ["mnist_noenc_backprop_test.py"],
    deps = [
        "//tf_shell_ml",
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "mnist_enc_backprop_test",
    size = "large",
    srcs = ["mnist_enc_backprop_test.py"],
    deps = [
        "//tf_shell_ml",
        requirement("tensorflow-cpu"),
    ],
)
