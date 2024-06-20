import tensorflow as tf
import tf_shell
import tf_shell.python.optimizers.shell_optimizers as shell_optimizers
import test_utils


# These test cases are for the CtPtOptimizer, which optimizes the tf graph by
# reordering ciphertext - plaintext operations which look like
# ((ct + pt) + pt) to (ct + (pt + pt).


@tf.function
def ct_pt_pt_add(ct, pt):
    return ((((((ct + pt) + pt) + pt) + pt) + pt) + pt) + pt


@tf.function
def ct_pt_pt_sub(ct, pt):
    return ((((((ct - pt) - pt) - pt) - pt) - pt) - pt) - pt


@tf.function
def ct_pt_pt_add_sub(ct, pt):
    return ((((((ct + pt) - pt) + pt) - pt) + pt) - pt) + pt


@tf.function
def ct_pt_pt_mul(ct, pt):
    return (ct * pt) * pt


@tf.function
def ct_pt_pt_add_mul_no_opt(ct, pt):
    # This should not be optimized, mul and add are not commutative.
    return (ct + pt) * pt


@tf.function
def ct_pt_pt_mul_add_no_opt(ct, pt):
    # This should not be optimized, mul and add are not commutative.
    return (ct * pt) + pt


def count_ct_pt_ops(graph, op_name):
    num_ct_pt_ops = 0
    for node in graph.as_graph_def().node:
        if node.op == op_name:
            num_ct_pt_ops += 1
    return num_ct_pt_ops


class TestCtPtOptimizer(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        int_dtypes = [
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.uint32,
            tf.int32,
            tf.uint64,
            tf.int64,
        ]
        cls.test_contexts = []

        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[1],
                plaintext_dtype=tf.float32,
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=1,
                mul_depth_supported=0,
            )
        )

    def _test_func(self, test_context, tf_func, num_pts, num_opt_pts, op_name):
        a = test_utils.uniform_for_n_muls(test_context, num_pts + 1)
        b = test_utils.uniform_for_n_muls(test_context, num_pts + 1)

        ct_a = tf_shell.to_encrypted(a, test_context.key, test_context.shell_context)
        pt_b = tf_shell.to_shell_plaintext(b, test_context.shell_context)

        # Sanity check the plain TensorFlow function correctly computes the
        # correct value.
        enc_c = tf_func(ct_a, pt_b)
        self.assertAllClose(
            tf_shell.to_tensorflow(enc_c, test_context.key),
            tf_func(a, b),
            atol=1 / test_context.shell_context.scaling_factor * num_pts,
        )

        func = tf_func.get_concrete_function(ct_a, pt_b)
        orig_num_ops = count_ct_pt_ops(func.graph, op_name)
        self.assertEqual(orig_num_ops, num_pts)

        # print("\noriginal graph:")
        # for node in func.graph.as_graph_def().node:
        #     print(f'{node.name} {node.op}({node.input})')

        # Optimize the graph using tf_shells HE-specific optimizers.
        optimized_func = shell_optimizers.optimize_shell_graph(func)
        # Call the optimized function.
        enc_c = optimized_func(ct_a, pt_b)
        # Can remove pack_output above if
        # https://github.com/tensorflow/tensorflow/pull/67612 is merged.
        enc_c = optimized_func.function_type.pack_output(enc_c)
        opt_num_ops = count_ct_pt_ops(optimized_func.graph, op_name)

        # print("\noptimized graph:")
        # for node in optimized_func.graph.as_graph_def().node:
        #     print(f'{node.name} {node.op}({node.input})')

        self.assertEqual(opt_num_ops, 1)

        # Check the optimized graph still computes the correct value.
        self.assertAllClose(
            tf_shell.to_tensorflow(enc_c, test_context.key),
            tf_func(a, b),
            atol=1 / test_context.shell_context.scaling_factor * num_pts,
        )

    def test_func(self):
        for test_context in self.test_contexts:
            with self.subTest(f"Optimizer for func ct_pt_pt_add."):
                self._test_func(test_context, ct_pt_pt_add, 7, 1, "AddCtPt64")
            with self.subTest(f"Optimizer for func ct_pt_pt_sub."):
                self._test_func(test_context, ct_pt_pt_sub, 7, 1, "SubCtPt64")
            with self.subTest(f"Optimizer for func ct_pt_pt_mul."):
                self._test_func(test_context, ct_pt_pt_mul, 2, 1, "MulCtPt64")
            with self.subTest(f"Optimizer for func ct_pt_pt_add_mul_no_opt."):
                self._test_func(
                    test_context, ct_pt_pt_add_mul_no_opt, 1, 1, "MulCtPt64"
                )
                self._test_func(
                    test_context, ct_pt_pt_add_mul_no_opt, 1, 1, "AddCtPt64"
                )
            with self.subTest(f"Optimizer for func ct_pt_pt_mul_add_no_opt."):
                self._test_func(
                    test_context, ct_pt_pt_mul_add_no_opt, 1, 1, "MulCtPt64"
                )
                self._test_func(
                    test_context, ct_pt_pt_mul_add_no_opt, 1, 1, "AddCtPt64"
                )


class TestCtPtAutoEnableOptimizer(tf.test.TestCase):
    def test_auto_optimize(self):
        from timeit import timeit

        context = tf_shell.create_context64(
            log_n=10,
            main_moduli=[8556589057, 8388812801],
            plaintext_modulus=40961,
            scaling_factor=3,
            mul_depth_supported=3,
            seed="test_seed",
        )

        secret_key = tf_shell.create_key64(context)

        a = tf.random.uniform([context.num_slots, 40000], dtype=tf.float32, maxval=10)
        b = tf.random.uniform([context.num_slots, 40000], dtype=tf.float32, maxval=10)

        ct_a = tf_shell.to_encrypted(a, secret_key, context)
        pt_b = tf_shell.to_shell_plaintext(b, context)

        # Call the function as usual.
        unopt_time = timeit(lambda: ct_pt_pt_add(ct_a, pt_b), number=1)

        # Turn on automatic optimization. Note there is no way to get the
        # optimized graph from the tf.function so we need to rely on timing info
        # to make sure it's turned on.
        shell_optimizers.enable_tf_shell_optimizer(["CtPtOptimizer"])

        opt_time = timeit(lambda: ct_pt_pt_add(ct_a, pt_b), number=1)

        # Optimized time should be twice as fast due to the two ciphertext
        # components, but give it some slack and check if it is 1.7x faster.
        self.assertLess(opt_time, unopt_time / 1.7)


if __name__ == "__main__":
    tf.test.main()
