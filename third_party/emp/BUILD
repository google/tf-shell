load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "emp-tool",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
        "CMAKE_INSTALL_PREFIX": "local",
    },
    lib_source = "@emp-tool//:all_srcs",
    out_data_dirs = [
        "cmake",
    ],
    out_shared_libs = [
        "libemp-tool.so",
    ],
)

cmake(
    name = "emp-ot",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
        "CMAKE_INSTALL_PREFIX": "./local",
        # CMake's find_package wants to find cmake config for liba,
        # which we do not have -> disable search
        # "CMAKE_DISABLE_FIND_PACKAGE_emp-tool": "True",
        # as currently we copy all libraries, built with Bazel, into $EXT_BUILD_DEPS/lib
        # and the headers into $EXT_BUILD_DEPS/include
        # "emp-tool_DIR": "$$EXT_BUILD_DEPS/emp-tool/",
        "EMP-TOOL_INCLUDE_DIR": "$$EXT_BUILD_DEPS/emp-tool/include",
        "EMP-TOOL_LIBRARY": "$$EXT_BUILD_DEPS/emp-tool/lib",
        # "EMP-TOOL_LIB": "$$EXT_BUILD_DEPS/emp-tool",
        # "EMP-TOOL_ROOT": "$$EXT_BUILD_DEPS/emp-tool",
        "CMAKE_INCLUDE_PATH": "$$EXT_BUILD_DEPS/emp-tool",
    },
    lib_source = "@emp-ot//:all_srcs",
    out_data_dirs = [
        "cmake",
    ],
    out_headers_only = True,
    deps = [
        ":emp-tool",
    ],
)

cmake(
    name = "emp-sh2pc",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
        "CMAKE_INSTALL_PREFIX": "./local",
        "CMAKE_INCLUDE_PATH": "$$EXT_BUILD_DEPS/emp-tool",
        "EMP-TOOL_INCLUDE_DIR": "$$EXT_BUILD_DEPS/emp-tool/include",
        "EMP-TOOL_LIBRARY": "$$EXT_BUILD_DEPS/emp-tool/lib/libemp-tool.so",
        "EMP-OT_INCLUDE_DIR": "$$EXT_BUILD_DEPS/emp-ot/include",
    },
    lib_source = "@emp-sh2pc//:all_srcs",
    out_data_dirs = [
        "cmake",
    ],
    out_headers_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":emp-ot",
        ":emp-tool",
    ],
)
