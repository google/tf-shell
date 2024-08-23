workspace(name = "tf-shell")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "shell_encryption",
    integrity = "sha256-6uWbertH7tjs/ef632lNy/M0ndM0LVVF8OZr+8CMNoI=",
    strip_prefix = "shell-encryption-300731a8c0e79e23289523d5ca405e004e749ef4",
    urls = ["https://github.com/google/shell-encryption/archive/300731a8c0e79e23289523d5ca405e004e749ef4.zip"],
)

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
