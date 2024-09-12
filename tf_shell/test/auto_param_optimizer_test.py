import tensorflow as tf
import tf_shell
import tf_shell.python.shell_optimizers as shell_optimizers


# These test cases are for the PtPtOptimizer, which optimizes the tf graph by
# reordering plaintext - plaintext operations which look like op( encode(a),
# <encode(b)>) to encode( op(a, <b>) ) where <> indicate optional arguments.

test_values_num_bits = 8


def gen_autocontext(cleartext_sz, noise_offset, scaling_factor=1):
    return tf_shell.create_autocontext64(
        log2_cleartext_sz=cleartext_sz,
        scaling_factor=scaling_factor,
        noise_offset_log2=noise_offset,
        noise_variance=8,
    )


def gen_context(scaling_factor=1):
    # Large moduli to avoid overflow in all test cases.
    return tf_shell.create_context64(
        log_n=11,
        main_moduli=[288230376151760897, 288230376152137729],
        plaintext_modulus=4294991873,
        scaling_factor=scaling_factor,
        seed="test_seed",
    )


@tf.function
def ct_ct_add(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits + 1, 32)
        if use_auto_context
        else gen_context()
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_encrypted(cleartext_b, key, shell_context)

    intermediate = a + b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_ct_mul(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2 + 1, 32)
        if use_auto_context
        else gen_context()
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_encrypted(cleartext_b, key, shell_context)

    intermediate = a * b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_pt_add(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2 + 1, 32)
        if use_auto_context
        else gen_context()
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = a + b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_pt_mul(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2 + 1, 32)
        if use_auto_context
        else gen_context()
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = a * b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


@tf.function
def long_arith(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2 + 3, 32)
        if use_auto_context
        else gen_context()
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = ((a * b) + a) + b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


@tf.function
def long_arith_with_scaling(cleartext_a, cleartext_b, use_auto_context=False):
    scaling_factor = 3
    shell_context = (
        gen_autocontext(test_values_num_bits * 2 + 3, 52, scaling_factor)
        if use_auto_context
        else gen_context(scaling_factor)
    )
    key = tf_shell.create_key64(shell_context)
    a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
    b = tf_shell.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = ((a * b) + a) + b
    result = tf_shell.to_tensorflow(intermediate, key)
    return result


# @tf.function
# def ct_roll(cleartext_a, cleartext_b, use_auto_context=False):
#     shell_context = (
#         gen_autocontext(test_values_num_bits)
#         if use_auto_context
#         else gen_context()
#     )
#     key = tf_shell.create_key64(shell_context)
#     a = tf_shell.to_encrypted(cleartext_a, key, shell_context)
#     b = tf_shell.to_shell_plaintext(cleartext_b, shell_context)

#     intermediate = (a * b) + b + a
#     result = tf_shell.to_tensorflow(intermediate, key)
#     return result


def count_ops(graph, op_name):
    num_ct_pt_ops = 0
    for node in graph.as_graph_def().node:
        if node.op == op_name:
            num_ct_pt_ops += 1
    return num_ct_pt_ops


class TestAutoParamOptimizer(tf.test.TestCase):

    def _test_func(self, tf_func):
        shape = [100, 12]
        a = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=2**test_values_num_bits - 1,
        )
        b = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=2**test_values_num_bits - 1,
        )
        a = tf.cast(a, tf.uint64)
        b = tf.cast(b, tf.uint64)

        func = tf_func.get_concrete_function(a, b, True)

        # print("\noriginal graph:")
        # for node in func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})")

        orig_num_auto_ops = count_ops(func.graph, "AutoShellContext64")
        self.assertEqual(orig_num_auto_ops, 1)

        # Optimize the graph using tf_shells HE-specific optimizers.
        optimized_func = shell_optimizers.optimize_shell_graph(
            func, ["ModuliAutotuneOptimizer"]
        )

        # print("\noptimized graph:")
        # for node in optimized_func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})")

        # Call the optimized function.
        c = optimized_func(a, b, True)
        # Can remove pack_output above if
        # https://github.com/tensorflow/tensorflow/pull/67612 is merged.
        c = optimized_func.function_type.pack_output(c)

        opt_num_auto_ops = count_ops(optimized_func.graph, "AutoShellContext64")
        self.assertEqual(opt_num_auto_ops, 0)
        opt_num_ctx_ops = count_ops(optimized_func.graph, "ContextImport64")
        self.assertEqual(opt_num_ctx_ops, 1)

        # Check the optimized graph still computes the correct value.
        eager_c = tf_func(a, b, False)
        padded_c = tf.pad(
            eager_c,
            [[0, c.shape[0] - eager_c.shape[0]]]
            + [[0, 0] for _ in range(len(c.shape) - 1)],
        )
        self.assertAllEqual(c, padded_c)

    def test_func(self):
        with self.subTest(f"Optimizer for func ct_ct_add."):
            self._test_func(ct_ct_add)

        with self.subTest(f"Optimizer for func ct_ct_mul."):
            self._test_func(ct_ct_mul)

        with self.subTest(f"Optimizer for func ct_pt_add."):
            self._test_func(ct_pt_add)

        with self.subTest(f"Optimizer for func ct_pt_mul."):
            self._test_func(ct_pt_mul)

        with self.subTest(f"Optimizer for func long_arith."):
            self._test_func(long_arith)

        with self.subTest(f"Optimizer for func long_arith_with_scaling."):
            self._test_func(long_arith_with_scaling)


class TestAutoParamEnableOptimizer(tf.test.TestCase):
    def test_func(self):
        shape = [20, 1]
        a = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=127,
        )
        b = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=127,
        )

        shell_optimizers.enable_tf_shell_optimizer()
        c = ct_ct_add(a, b, True)

        # If the optimizer ran, the shape should be padded out to the
        # ciphertext modulus.
        self.assertNotEqual(c.shape, shape)


if __name__ == "__main__":
    tf.test.main()
