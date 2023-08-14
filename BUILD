load("//:version.bzl", "VERSION_LABEL")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:packaging.bzl", "py_wheel")
load("@buildifier_prebuilt//:rules.bzl", "buildifier")

exports_files([
    "LICENSE",
    "setup.py",
    "requirements.in",
    "requirements.txt",
    "README.md",
    "DESCRIPTION.md",
])

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements.txt",
    visibility = ["//visibility:public"],
)

buildifier(
    name = "bazel_formatter",
    exclude_patterns = [
        "./bazel-*/*",
        "./.git/*",
    ],
    lint_mode = "warn",
)

py_binary(
    name = "python_formatter",
    srcs = ["//tools:python_formatter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
    deps = [
        requirement("black"),
    ],
)

py_binary(
    name = "clang_formatter",
    srcs = ["//tools:clang_formatter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
)

py_wheel(
    name = "wheel",
    abi = "ABI",
    author = "Google Inc.",
    author_email = "jchoncholas@google.com",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Security :: Cryptography",
    ],
    description_file = "//:DESCRIPTION.md",
    distribution = "tf-shell",
    extra_distinfo_files = {
        "//:LICENSE": "LICENSE",
        "//:README.md": "README",
    },
    homepage = "https://github.com/google/tf-shell",
    license = "Apache 2.0",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:windows_arm64": "win_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_12_0_x86_64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_12_0_arm64",
        "@bazel_tools//src/conditions:linux_x86_64": "LINUX_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "LINUX_aarch64",
    }),
    python_requires = ">=3.8.0",
    python_tag = "INTERPRETER",
    requires = ["tensorflow-cpu==2.13.0"],
    version = VERSION_LABEL,
    deps = [
        "//shell_ml:shell_ml_pkg",
        "//shell_tensor:shell_tensor_pkg",
    ],
)
