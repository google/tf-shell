#!/usr/bin/python
#
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from pybind11_abseil.pybind11_abseil import status
import shell_tensor.shell as shell

# Bindings tests merely exercise that library functions are exposed, but do not
# additionally verify functionality, since that is the responsibility of the
# underlying SHELL library.


class TestContextBindings(unittest.TestCase):
    def test_context_constructor_exception(self):
        with self.assertRaises(TypeError):
            shell.ContextParams64()  # needs args

    def test_context_exception(self):
        params = shell.ContextParams64(modulus=4, log_n=1, log_t=1, variance=1)
        with self.assertRaises(status.StatusNotOk):
            shell.Context64.Create(params)  # modulus should be odd

    def test_shell_context64(self):
        params = shell.ContextParams64(
            modulus=shell.kModulus59, log_n=10, log_t=11, variance=8
        )
        ctx = shell.Context64.Create(params)
        self.assertEqual(ctx.GetModulus(), shell.kModulus59)
        self.assertEqual(ctx.GetLogN(), 10)
        self.assertEqual(ctx.GetLogT(), 11)
        self.assertEqual(ctx.GetVariance(), 8)

        mod_params = ctx.GetModulusParams()
        self.assertEqual(mod_params.modulus, shell.kModulus59)

        ntt_params = ctx.GetNttParams()
        self.assertEqual(ntt_params.number_coeffs, 1024)

        err_params = ctx.GetErrorParams()
        self.assertAlmostEqual(err_params.B_scale(), 9804626.934608372)


class TestPrngBindings(unittest.TestCase):
    def test_no_seed_exception(self):
        with self.assertRaises(TypeError):
            shell.SingleThreadHkdfPrng.Create()

    def test_bad_seed_exception(self):
        bad_seed = "too short"
        with self.assertRaises(status.StatusNotOk):
            shell.SingleThreadHkdfPrng.Create(bad_seed)

    def test_good_seed(self):
        seed = shell.SingleThreadHkdfPrng.GenerateSeed()
        self.assertEqual(len(seed), shell.SingleThreadHkdfPrng.SeedLength())

        prng = shell.SingleThreadHkdfPrng.Create(seed)
        r8 = prng.Rand8()
        r64 = prng.Rand64()


class TestPrimitiveBindings(unittest.TestCase):
    def test_montgomery(self):
        params = shell.ContextParams64(
            modulus=shell.kModulus59, log_n=10, log_t=11, variance=8
        )
        context = shell.Context64.Create(params)
        mont_int = shell.MontgomeryInt64.ImportInt(27, context.GetModulusParams())

    def test_polynomial(self):
        params = shell.ContextParams64(
            modulus=shell.kModulus59, log_n=10, log_t=11, variance=8
        )
        context = shell.Context64.Create(params)
        poly = shell.Polynomial64(2**7, context.GetModulusParams())
        poly2 = shell.Polynomial64(2**7, context.GetModulusParams())


class TestSymmetricBindings(unittest.TestCase):
    def test_sample_key_exception(self):
        with self.assertRaises(TypeError):
            shell.SingleThreadHkdfPrng.Create()

    def test_polynomial_encrypt(self):
        params = shell.ContextParams64(
            modulus=shell.kModulus59, log_n=10, log_t=11, variance=8
        )
        context = shell.Context64.Create(params)
        num_slots = 2 ** context.GetLogN()

        seed = shell.SingleThreadHkdfPrng.GenerateSeed()
        prng = shell.SingleThreadHkdfPrng.Create(seed)

        key = shell.SymmetricKey64.Sample(context, prng)

        vect = shell.VectorInt64(num_slots)
        for i in range(num_slots):
            vect[i] = i
        mvect = shell.VectorMontgomeryInt64.ImportVect(vect, context.GetModulusParams())

        poly = shell.Polynomial64.ConvertToNtt(
            mvect, context.GetNttParams(), context.GetModulusParams()
        )

        ct = shell.SymmetricCt64.Encrypt(key, poly, context, prng)
        pt = shell.SymmetricCt64.Decrypt(key, ct)

        for i in range(num_slots):
            self.assertEqual(pt[i], i)

    def test_encrypt(self):
        params = shell.ContextParams64(
            modulus=shell.kModulus59, log_n=10, log_t=11, variance=8
        )
        context = shell.Context64.Create(params)
        num_slots = 2 ** context.GetLogN()

        seed = shell.SingleThreadHkdfPrng.GenerateSeed()
        prng = shell.SingleThreadHkdfPrng.Create(seed)

        key = shell.SymmetricKey64.Sample(context, prng)

        vect = shell.VectorInt64(num_slots)
        for i in range(num_slots):
            vect[i] = i

        ct = shell.SymmetricCt64.Encrypt(key, vect, context, prng)
        pt = shell.SymmetricCt64.Decrypt(key, ct)

        for i in range(num_slots):
            self.assertEqual(pt[i], i)


if __name__ == "__main__":
    unittest.main()
