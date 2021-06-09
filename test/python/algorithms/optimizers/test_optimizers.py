# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Optimizers """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from ddt import ddt, data, unpack
import numpy as np
from scipy.optimize import rosen, rosen_der

from qiskit.algorithms.optimizers import (
    ADAM,
    AQGD,
    BOBYQA,
    IMFIL,
    CG,
    COBYLA,
    GSLS,
    GradientDescent,
    L_BFGS_B,
    NELDER_MEAD,
    Optimizer,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA,
    QNSPSA,
    TNC,
    SciPyOptimizer,
)
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.utils import algorithm_globals

try:
    import skquant.opt as skq  # pylint: disable=unused-import

    _HAS_SKQUANT = True
except ImportError:
    _HAS_SKQUANT = False


class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test Optimizers"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52

    def _optimize(self, optimizer, grad=False):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        if grad:
            res = optimizer.optimize(
                len(x_0), rosen, gradient_function=rosen_der, initial_point=x_0
            )
        else:
            res = optimizer.optimize(len(x_0), rosen, initial_point=x_0)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x_0), decimal=2)
        return res

    def test_adam(self):
        """adam test"""
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cg(self):
        """cg test"""
        optimizer = CG(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_gradient_descent(self):
        """cg test"""
        optimizer = GradientDescent(maxiter=100000, tol=1e-06, learning_rate=1e-3)
        res = self._optimize(optimizer, grad=True)
        self.assertLessEqual(res[2], 100000)

    def test_cobyla(self):
        """cobyla test"""
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_l_bfgs_b(self):
        """l_bfgs_b test"""
        optimizer = L_BFGS_B(maxfun=1000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_p_bfgs(self):
        """parallel l_bfgs_b test"""
        optimizer = P_BFGS(maxfun=1000, max_processes=4)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_nelder_mead(self):
        """nelder mead test"""
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_powell(self):
        """powell test"""
        optimizer = POWELL(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_slsqp(self):
        """slsqp test"""
        optimizer = SLSQP(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        """spsa test"""
        optimizer = SPSA(maxiter=10000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_tnc(self):
        """tnc test"""
        optimizer = TNC(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_gsls(self):
        """gsls test"""
        optimizer = GSLS(
            sample_size_factor=40,
            sampling_radius=1.0e-12,
            maxiter=10000,
            max_eval=10000,
            min_step_size=1.0e-12,
        )
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        _, x_value, n_evals = optimizer.optimize(len(x_0), rosen, initial_point=x_0)

        # Ensure value is near-optimal
        self.assertLessEqual(x_value, 0.01)
        self.assertLessEqual(n_evals, 10000)

    def test_scipy_optimizer(self):
        """scipy_optimizer test"""
        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_scipy_optimizer_callback(self):
        """scipy_optimizer callback test"""
        values = []

        def callback(x):
            values.append(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1000}, callback=callback)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)
        self.assertTrue(values)  # Check the list is nonempty.


@ddt
class TestOptimizerSerialization(QiskitAlgorithmsTestCase):
    """Tests concerning the serialization of optimizers."""

    @data(
        ("BFGS", {"maxiter": 100, "eps": np.array([0.1])}),
        ("CG", {"maxiter": 100, "gtol": 1e-8}),
        ("COBYLA", {"maxiter": 100}),
        ("L_BFGS_B", {"maxiter": 100}),
        ("NELDER_MEAD", {"maxiter": 100}),
        ("NFT", {"maxiter": 100}),
        ("P_BFGS", {"maxiter": 100}),
        ("POWELL", {"maxiter": 100}),
        ("SLSQP", {"maxiter": 100}),
        ("TNC", {"maxiter": 100}),
        ("dogleg", {"maxiter": 100}),
        ("trust-constr", {"maxiter": 100}),
        ("trust-ncg", {"maxiter": 100}),
        ("trust-exact", {"maxiter": 100}),
        ("trust-krylov", {"maxiter": 100}),
    )
    @unpack
    def test_scipy(self, method, options):
        """Test the SciPyOptimizer is serializable."""

        optimizer = SciPyOptimizer(method, options=options)
        serialized = optimizer.to_dict()
        from_dict = SciPyOptimizer.from_dict(serialized)

        self.assertEqual(from_dict._method, method.lower())
        self.assertEqual(from_dict._options, options)

    def test_scipy_not_serializable(self):
        """Test serialization fails if the optimizer contains an attribute that's not supported."""

        def callback(x):
            print(x)

        optimizer = SciPyOptimizer("BFGS", options={"maxiter": 1}, callback=callback)

        with self.assertRaises(QiskitError):
            _ = optimizer.to_dict()

    def test_scipy_name_missing(self):
        """Test serialization fails if the dictionary has no 'name' key."""
        with self.assertRaises(ValueError):
            _ = SciPyOptimizer.from_dict({"maxiter": 1})

    def test_adam(self):
        """Test ADAM is serializable."""

        adam = ADAM(maxiter=100, amsgrad=True)
        serialized = adam.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, ADAM)
        self.assertEqual(reconstructed._maxiter, 100)
        self.assertTrue(reconstructed._amsgrad)

    def test_aqgd(self):
        """Test AQGD is serializable."""

        opt = AQGD(maxiter=[200, 100], eta=[0.2, 0.1], momentum=[0.25, 0.1])
        serialized = opt.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, AQGD)
        self.assertListEqual(reconstructed._maxiter, [200, 100])
        self.assertListEqual(reconstructed._eta, [0.2, 0.1])
        self.assertListEqual(reconstructed._momenta_coeff, [0.25, 0.1])

    @unittest.skipIf(not _HAS_SKQUANT, "Install scikit-quant to run this test.")
    def test_bobyqa(self):
        """Test BOBYQA is serializable."""

        opt = BOBYQA(maxiter=200)
        serialized = opt.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, BOBYQA)
        self.assertEqual(reconstructed._maxiter, 200)

    @unittest.skipIf(not _HAS_SKQUANT, "Install scikit-quant to run this test.")
    def test_imfil(self):
        """Test IMFIL is serializable."""

        opt = IMFIL(maxiter=200)
        serialized = opt.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, IMFIL)
        self.assertEqual(reconstructed._maxiter, 200)

    def test_gradient_descent(self):
        """Test GradientDescent is serializable."""

        opt = GradientDescent(maxiter=10, learning_rate=0.01)
        serialized = opt.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, GradientDescent)
        self.assertEqual(reconstructed.maxiter, 10)
        self.assertEqual(reconstructed.learning_rate, 0.01)

    def test_gsls(self):
        """Test GSLS is serializable."""

        opt = GSLS(maxiter=100, sampling_radius=1e-3)
        serialized = opt.to_dict()

        reconstructed = Optimizer.from_dict(serialized)
        self.assertIsInstance(reconstructed, GSLS)
        self.assertEqual(reconstructed._options["maxiter"], 100)
        self.assertTrue(reconstructed._options["sampling_radius"], 1e-3)

    def test_spsa(self):
        """Test SPSA optimizer is serializable."""
        options = {
            "maxiter": 100,
            "blocking": True,
            "allowed_increase": 0.1,
            "second_order": True,
            "learning_rate": 0.02,
            "perturbation": 0.05,
            "regularization": 0.1,
            "resamplings": 2,
            "perturbation_dims": 5,
            "trust_region": False,
            "initial_hessian": None,
            "hessian_delay": 0,
        }
        spsa = SPSA(**options)

        serialized = spsa.to_dict()
        expected = options.copy()
        expected["name"] = "SPSA"

        with self.subTest(msg="check constructed dictionary"):
            self.assertDictEqual(serialized, expected)

        reconstructed = Optimizer.from_dict(serialized)
        with self.subTest(msg="test reconstructed optimizer"):
            self.assertDictEqual(reconstructed.to_dict(), expected)

    def test_spsa_custom_iterators(self):
        """Test serialization works with custom iterators for learning rate and perturbation."""
        rate = 0.99

        def powerlaw():
            n = 0
            while True:
                yield rate ** n
                n += 1

        def steps():
            n = 1
            divide_after = 20
            epsilon = 0.5
            while True:
                yield epsilon
                n += 1
                if n % divide_after == 0:
                    epsilon /= 2

        learning_rate = powerlaw()
        expected_learning_rate = np.array([next(learning_rate) for _ in range(200)])

        perturbation = steps()
        expected_perturbation = np.array([next(perturbation) for _ in range(200)])

        spsa = SPSA(maxiter=200, learning_rate=powerlaw, perturbation=steps)
        serialized = spsa.to_dict()

        self.assertTrue(np.allclose(serialized["learning_rate"], expected_learning_rate))
        self.assertTrue(np.allclose(serialized["perturbation"], expected_perturbation))

    def test_qnspsa(self):
        """Test QN-SPSA optimizer is serializable."""
        ansatz = RealAmplitudes(1)
        fidelity = QNSPSA.get_fidelity(ansatz)
        options = {
            "fidelity": fidelity,
            "maxiter": 100,
            "blocking": True,
            "allowed_increase": 0.1,
            "learning_rate": 0.02,
            "perturbation": 0.05,
            "regularization": 0.1,
            "resamplings": 2,
            "perturbation_dims": 5,
            "initial_hessian": None,
            "hessian_delay": 0,
        }
        spsa = QNSPSA(**options)

        serialized = spsa.to_dict()
        expected = options.copy()
        expected.pop("fidelity")  # fidelity cannot be serialized
        expected["name"] = "QNSPSA"

        with self.subTest(msg="check constructed dictionary"):
            self.assertDictEqual(serialized, expected)

        # fidelity cannot be serialized, so it must be added back in
        serialized["fidelity"] = fidelity
        reconstructed = Optimizer.from_dict(serialized)
        with self.subTest(msg="test reconstructed optimizer"):
            self.assertDictEqual(reconstructed.to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
