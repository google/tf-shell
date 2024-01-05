workspace(name = "tf_shell")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# requires pybind >= 2.11.0
git_repository(
    name = "pybind11_abseil",
    commit = "67491a4176dad18d5971047c40ae12ca355d4fcb",
    remote = "https://github.com/pybind/pybind11_abseil.git",
)

http_archive(
    name = "shell_encryption",
    sha256 = "894373a7c369b8b7b01e9a2e98a18b801a9f318d261988faaf856dbcf099a8a4",
    strip_prefix = "shell-encryption-a0b2e04616928b8c3d9be581cc7680032d44eaf8",
    urls = ["https://github.com/google/shell-encryption/archive/a0b2e04616928b8c3d9be581cc7680032d44eaf8.zip"],
)

http_archive(
    name = "com_github_tink_crypto_tink_cc",
    sha256 = "103ddfce800e77f3b3b6b2c808a8611bc734b31ddb12fbcfd8bebc1b96a7e963",
    strip_prefix = "tink-cc-2.0.0",
    urls = ["https://github.com/tink-crypto/tink-cc/archive/refs/tags/v2.0.0.zip"],
)

load("@com_github_tink_crypto_tink_cc//:tink_cc_deps.bzl", "tink_cc_deps")

tink_cc_deps()

load("@com_github_tink_crypto_tink_cc//:tink_cc_deps_init.bzl", "tink_cc_deps_init")

tink_cc_deps_init()

# rules_cc defines rules for generating C++ code from Protocol Buffers.
http_archive(
    name = "rules_cc",
    sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
    strip_prefix = "rules_cc-0.0.9",
    urls = [
        "https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

http_archive(
    name = "rules_proto",
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

http_archive(
    name = "com_github_google_googletest",
    sha256 = "755f9a39bc7205f5a0c428e920ddad092c33c8a1b46997def3f1d4a82aded6e1",
    strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
)

# must be > 20230125.1 due to https://github.com/abseil/abseil-cpp/pull/1399
http_archive(
    name = "com_google_absl",
    sha256 = "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    strip_prefix = "abseil-cpp-20230802.1",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.1.tar.gz",
    ],
)

git_repository(
    name = "boringssl",
    commit = "67ffb9606462a1897d3a5edf5c06d329878ba600",  # https://boringssl.googlesource.com/boringssl/+/refs/heads/master-with-bazel
    patch_args = ["-p1"],
    patches = ["//tools/0001-Posix-std.patch"],  # git format-patch --keep-subject --no-stat --zero-commit <commit>
    remote = "https://boringssl.googlesource.com/boringssl",
    shallow_since = "1585767053 +0000",
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "6281aa4eeecb9e932d7091f99872e7b26fa6aacece49c15ce5b14af2b7ec050f",
    strip_prefix = "glog-96a2f23dca4cc7180821ca5f32e526314395d26a",
    urls = ["https://github.com/google/glog/archive/96a2f23dca4cc7180821ca5f32e526314395d26a.zip"],
)

# gflags, needed for glog
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)
