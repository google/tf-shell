"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import sys


def main():
    bazel_dir_var = "BUILD_WORKSPACE_DIRECTORY"
    if not bazel_dir_var in os.environ:
        raise Exception(f"{bazel_dir_var} not defined")
    build_workspace_dir = os.environ[bazel_dir_var]

    found_files = []

    extensions = (".c", ".cc", ".h", ".hh")
    exclude = [".venv", ".git", "third_party", "bazel-*"]

    for root, _, files in os.walk(build_workspace_dir):
        for file in files:
            if file.endswith(extensions) and not any(x in root for x in exclude):
                found_files.append(os.path.join(root, file))

    format_command = "clang-format --style=file -i " + " ".join(found_files)
    # print(f"Running clang-format with command: {format_command}")

    try:
        formatted = subprocess.call(format_command, shell=True)
    except OSError:
        print("Running clang-format failed with non-zero status.", file=sys.stderr)
        print(f"Command    : {format_command}", file=sys.stderr)
        if formatted is not None:
            print(f"Return code: {formatted}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
