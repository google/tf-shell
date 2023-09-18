workspace(name = "tf_shell")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# requires pybind >= 2.11.0
git_repository(
    name = "pybind11_abseil",
    commit = "215bcc601cf2e373f59444a1ad837a353bd20696",
    remote = "https://github.com/pybind/pybind11_abseil.git",
)

http_archive(
    name = "shell_encryption",
    sha256 = "8b45d1e511c4f259e67205aa9ea45176a866b704346ef346b15a783d11fe2567",
    strip_prefix = "shell-encryption-f89f60e112d74629d96bb5714500fafa5649d338",
    urls = ["https://github.com/google/shell-encryption/archive/f89f60e112d74629d96bb5714500fafa5649d338.zip"],
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

http_archive(
    name = "rules_cc",
    sha256 = "35f2fb4ea0b3e61ad64a369de284e4fbbdcdba71836a5555abb5e194cf119509",
    strip_prefix = "rules_cc-624b5d59dfb45672d4239422fa1e3de1822ee110",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

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

http_archive(
    name = "com_github_google_googletest",
    sha256 = "755f9a39bc7205f5a0c428e920ddad092c33c8a1b46997def3f1d4a82aded6e1",
    strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
)

# must be > 20230125.1 due to https://github.com/abseil/abseil-cpp/pull/1399
http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-20230125.3",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/20230125.3.tar.gz",
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

http_archive(
    name = "rules_foreign_cc",
    sha256 = "d54742ffbdc6924f222d2179f0e10e911c5c659c4ae74158e9fe827aad862ac6",
    strip_prefix = "rules_foreign_cc-0.2.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.2.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

http_archive(
    name = "com_gnu_gmp",
    build_file = "@//shell_encryption:BUILD.gmp.bazel",
    sha256 = "fd4829912cddd12f84181c3451cc752be224643e87fac497b69edddadc49b4f2",
    strip_prefix = "gmp-6.2.1",
    urls = [
        "https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz",
        "https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz",
    ],
)
