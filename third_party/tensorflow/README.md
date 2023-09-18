# Linking to Tensorflow in Bazel

Targets elsewhere in this project need to link to tensorflow's DSOs and header
files. These tensorflow dependencies are version specific because they link
with cpython whose ABI changes for minor python versions e.g. 3.9 and 3.10.

The tensorflow custom ops example
[repo](https://github.com/tensorflow/custom-op) suggests installing tensorflow
on the host machine and compiling against those DSOs/headers. This causes
problems as the Bazel python rules are moving away from allowing use of the
system python interpreter, (see
[here](https://github.com/bazelbuild/rules_python/commit/0642390d387ac70e44ee794cc9c6dcf182762ad3)).
This means that there is a mismatch between the version of tensorflow installed
locally, and the version Bazel will use.

This directory contains targets which copy the DSOs and headers out of Bazel's
pip-installed tensorflow, and make them available to other targets to build
against. Ideally these would be symlinks, but for now this works.
