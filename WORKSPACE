workspace(name = "tf-shell")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "shell_encryption",
    integrity = "sha256-6uWbertH7tjs/ef632lNy/M0ndM0LVVF8OZr+8CMNoI=",
    strip_prefix = "shell-encryption-300731a8c0e79e23289523d5ca405e004e749ef4",
    urls = ["https://github.com/google/shell-encryption/archive/300731a8c0e79e23289523d5ca405e004e749ef4.zip"],
)

_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

http_archive(
    name = "emp-tool",
    build_file_content = _ALL_CONTENT,
    integrity = "sha256-jEpsiBYbcu1pqP9NYmIBresj6I83XeGb1IuMs0psI7c=",
    strip_prefix = "emp-tool-0.2.5",
    urls = ["https://github.com/emp-toolkit/emp-tool/archive/refs/tags/0.2.5.zip"],
)

http_archive(
    name = "emp-ot",
    build_file_content = _ALL_CONTENT,
    integrity = "sha256-wXi5/rDgw9EysMutJ4Tm1wirf/SGkDSJwsHQdeuA21E=",
    strip_prefix = "emp-ot-0.2.4",
    urls = ["https://github.com/emp-toolkit/emp-ot/archive/refs/tags/0.2.4.zip"],
)

http_archive(
    name = "emp-sh2pc",
    build_file_content = _ALL_CONTENT,
    integrity = "sha256-yQPfQV4sILqQcz/K5hWGL/eDePanilLChGIp3wVx8TI=",
    strip_prefix = "emp-sh2pc-0.2.2",
    urls = ["https://github.com/emp-toolkit/emp-sh2pc/archive/refs/tags/0.2.2.zip"],
)
