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
import black
import sys

bazel_build_dir = "BUILD_WORKSPACE_DIRECTORY"

if __name__ == "__main__":
    if not bazel_build_dir in os.environ:
        raise Exception(f"{bazel_build_dir} not defined")
    build_workspace_dir = os.environ[bazel_build_dir]

    sys.exit(
        black.main(
            sys.argv
            + [
                "--color",
                build_workspace_dir,
            ]
        )
    )
