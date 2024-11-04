# tf-shell

The `tf-shell` library supports privacy preserving machine learning with
homomorphic encryption via the
[SHELL](https://github.com/google/shell-encryption/) library and tensorflow.

This is not an officially supported Google product.

## Getting Started

```bash
pip install tf-shell
```

See `./examples/` for how to use the library.

## Background

Homomorphic encryption allows computation on encrypted data. For example, given
two ciphertexts `a` and `b` representing the numbers `3` and `4`, respectively,
one can compute a ciphertext `c` representing the number `7` without decrypting
`a` or `b`. This is useful for privacy preserving machine learning because it
allows training a model on encrypted data.

The SHELL encryption library supports homomorphic encryption with respect to
addition and multiplication. This means that one can compute the sum of two
ciphertexts or the product of two ciphertexts without decrypting them. SHELL
does not support _fully_ homomorphic encryption, meaning computing functions of
ciphertexts with arbitrary depth. That said, because machine learning models are
of bounded depth, the performance benefits of leveled schemes (without
bootstrapping, e.g. SHELL) outweight limitations in circuit depth.

## Design

This library has two modules, `tf_shell` which supports Tensorflow Tensors
containing ciphertexts with homomorphic properties, and `tf_shell_ml` some (very)
simple machine learning tools supporting privacy preserving training.

`tf-shell` is designed for Label-DP SGD where training data is vertically
partitioned, e.g. one party holds features while another party holds labels. The
party who holds the features would like to train a model without learning the
labels. The resultant trained model is differentially private with respect to
the labels.

## Building

### Build From Source

1. Install bazel and python3 or use the devcontainer.

2. Run the tests.

    ```bash
    bazel test //tf_shell/...
    bazel test //tf_shell_ml/...  # Large tests, requires 128GB of memory.
    ```

3. Build the code.

    ```bash
    bazel build //:wheel
    bazel run //:wheel_rename
    ```

4. (Optional) Install the wheel, e.g. to try out the `./examples/`.
    You may first need to copy the wheel out of the devcontainer's filesystem.

    ```bash
    cp -f bazel-bin/*.whl ./  # Run in devcontainer if using.
    ```

    Then install.

    ```bash
    pip install --force-reinstall tf_shell-*.whl  # Run in target environment.
    ```

Note the cpython api is not compatible across minor python versions (e.g. 3.10,
3.11) so the wheel must be rebuilt for each python version.

### Code Formatters and Counters

```bash
bazel run //:bazel_formatter
bazel run //:python_formatter
bazel run //:clang_formatter
```

```bash
cloc ./ --fullpath --not-match-d='/(bazel-.*|.*\.venv)/'
```

### Update Python Dependencies

Update requirements.in and run the following to update the requirements files
for each python version.

```bash
for ver in 3_9 3_10 3_11 3_12; do
  rm requirements_${ver}.txt
  touch requirements_${ver}.txt
  bazel run //:requirements_${ver}.update
done

bazel clean --expunge
```

If updating the tensorflow dependency, other dependencies may also need to
change, e.g. abseil (see `MODULE.bazel`). This issue usually manifests as a
missing symbols error in the tests when trying to import the tensorflow DSO. In
this case, `c++filt` will help to decode the mangled symbol name and `nm
--defined-only .../libtensorflow_framework.so | grep ...` may help find what the
symbol changed to, and which dependency is causing the error.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
