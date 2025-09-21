load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

exports_files([
    "python_formatter.py",
    "clang_formatter.py",
    "wheel_rename.py",
    "security_scan.py",
    "0001-Posix-std.patch",
])

py_binary(
    name = "security_scanner",
    srcs = ["security_scan.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
    # Note: Security scanning tools need to be installed separately
    # pip install -r requirements-security.txt
)
