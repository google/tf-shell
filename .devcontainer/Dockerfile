FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# Install Bazel
ARG BAZELISK_VERSION=v1.10.1
ARG BAZELISK_DOWNLOAD_SHA=dev-mode
RUN curl -fSsL -o /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && ([ "${BAZELISK_DOWNLOAD_SHA}" = "dev-mode" ] || echo "${BAZELISK_DOWNLOAD_SHA} */usr/local/bin/bazelisk" | sha256sum --check - ) \
    && chmod 0755 /usr/local/bin/bazelisk

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y python3 python3-pip python-is-python3 clang-format python3-venv cloc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts
