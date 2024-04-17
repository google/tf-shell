workspace(name = "tf-shell")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "pybind11_abseil",
    sha256 = "1496b112e86416e2dcf288569a3e7b64f3537f0b18132224f492266e9ff76c44",
    strip_prefix = "pybind11_abseil-202402.0",
    urls = ["https://github.com/pybind/pybind11_abseil/archive/refs/tags/v202402.0.tar.gz"],
)

http_archive(
    name = "shell_encryption",
    integrity = "sha256-6uWbertH7tjs/ef632lNy/M0ndM0LVVF8OZr+8CMNoI=",
    strip_prefix = "shell-encryption-300731a8c0e79e23289523d5ca405e004e749ef4",
    urls = ["https://github.com/google/shell-encryption/archive/300731a8c0e79e23289523d5ca405e004e749ef4.zip"],
)

http_archive(
    name = "com_github_tink_crypto_tink_cc",
    sha256 = "d0fefc61e3bde758c8773f1348e6a64fc4fd6ecafe62c4adc0df8957ce800757",
    strip_prefix = "tink-cc-2.1.2",
    urls = ["https://github.com/tink-crypto/tink-cc/archive/refs/tags/v2.1.2.zip"],
)

load("@com_github_tink_crypto_tink_cc//:tink_cc_deps.bzl", "tink_cc_deps")

tink_cc_deps()

load("@com_github_tink_crypto_tink_cc//:tink_cc_deps_init.bzl", "tink_cc_deps_init")

tink_cc_deps_init()

http_archive(
    name = "rules_proto",
    sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
    strip_prefix = "rules_proto-5.3.0-21.7",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
