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
This library has two modules, `shell_tensor` which supports Tensorflow Tensors
containing ciphertexts with homomorphic properties, and `shell_ml` some (very)
simple machine learning tools supporting privacy preserving training.

`tf-shell` was designed for Label-DP SGD where training data is vertically
partitioned, e.g. one party holds features while another party holds labels. The
party who holds the features would like to train a model without learning the
labels. The resultant trained model is differentially private with respect to
the labels.


## Building


### Build From Source
1. Install Bazelisk and python3 or use the devcontainer

2. Install requriements
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    echo "build --enable_bzlmod" >> .bazelrc
    [ -f ./requirements.txt ] || ( touch requirements.txt && bazelisk run //:requirements.update )
    pip install --require-hashes --no-deps -r requirements.txt
    ```

3. Setup tensorflow
    ```bash
    export TF_NEED_CUDA=0
    export PIP_MANYLINUX2010=1
    export TF_CUDA_VERSION=10.1
    ./configure_bazelrc.sh
    ```

3. Build
    ```bash
    bazelisk build --config release //:wheel
    python tools/wheel_rename.py
    cp -f bazel-bin/*-*-cp*.whl ./ # copy out of devcontainer fs
    ```

3. Install build artifact in a new venv, e.g. to try out the `./examples/`.
    ```bash
    pip install bazel-bin/tf-shell-...-manylinux.whl
    ```

Note the cpython api is not compatible across minor python versions (e.g. 3.10,
3.11) so the wheel must be rebuilt for each python version.


### Code Formatters and Counters
```bash
bazelisk run //:bazel_formatter
bazelisk run //:python_formatter
bazelisk run //:clang_formatter
```

```bash
cloc ./ --fullpath --not-match-d='/(bazel-.*|.*\.venv)/'
```


### Update Python Dependencies
```bash
bazelisk run //:requirements.update
```


### PyPI Package
```bash
bazelisk build -c opt //:wheel
python wheel_rename.py
export TWINE_USERNAME
export TWINE_PASSWORD
export TWINE_REPOSITORY
export TWINE_REPOSITORY_URL
export TWINE_CERT
twine upload
```


## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.


## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.


## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
