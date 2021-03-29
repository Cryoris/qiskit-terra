# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SPSA optimizer."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from qiskit import BasicAer
from qiskit.algorithms.optimizers._spsa import SPSA
from qiskit.algorithms.optimizers._qnspsa import QNSPSA
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliTwoDesign
from qiskit.utils import algorithm_globals
from qiskit.opflow import PauliSumOp, I, Z, Y, StateFn, CircuitSampler
from qiskit.algorithms import VQE
from qiskit.algorithms.minimum_eigen_solvers.qnspsa_vqe import QNSPSAVQE


class TestOptimizerQNSPSA(QiskitAlgorithmsTestCase):
    """ Test AQGD optimizer using RY for analytic gradient with VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.qubit_op = PauliSumOp.from_list([
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156),
        ])

    def get_overlap(self, circuit, backend):
        """Get the overlap function."""
        params_x = ParameterVector('x', circuit.num_parameters)
        params_y = ParameterVector('y', circuit.num_parameters)

        expression = ~StateFn(circuit.assign_parameters(
            params_x)) @ StateFn(circuit.assign_parameters(params_y))

        sampler = CircuitSampler(backend)

        def overlap_fn(values_x, values_y):
            value_dict = dict(zip(params_x[:] + params_y[:], values_x.tolist() + values_y.tolist()))
            return -0.5 * np.abs(sampler.convert(expression, params=value_dict).eval()) ** 2

        return overlap_fn

    def test_pauli_two_design(self):
        """Test the Pauli Two-design circuit."""
        backend = BasicAer.get_backend('statevector_simulator')
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        observable = I ^ Z ^ Z
        overlap_fn = self.get_overlap(circuit, backend)

        spsa = QNSPSA(overlap_fn,
                      hessian_resamplings=3,
                      allowed_increase=0.1,
                      maxiter=300,
                      blocking=True,
                      learning_rate=1e-2,
                      perturbation=1e-2)

        initial_point = np.random.random(circuit.num_parameters)
        vqe = VQE(circuit, spsa, initial_point, quantum_instance=backend)
        result = vqe.compute_minimum_eigenvalue(observable)
        print(result)

    def test_pauli_two_design_batches(self):
        """Test the Pauli Two-design circuit."""
        backend = BasicAer.get_backend('statevector_simulator')
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        observable = I ^ Z ^ Z
        overlap_fn = circuit

        initial_point = np.random.random(circuit.num_parameters)
        # result = spsa.optimize(circuit.num_parameters, obj, initial_point=initial_point)
        vqe = QNSPSAVQE(circuit, initial_point, quantum_instance=backend)
        result = vqe.compute_minimum_eigenvalue(observable)
        print(result)

    def test_pennylane(self):
        """Test the PennyLane example."""
        backend = BasicAer.get_backend('statevector_simulator')

        params = ParameterVector('x', 4)
        circuit = QuantumCircuit(3)
        circuit.ry(np.pi / 4, 0)
        circuit.ry(np.pi / 3, 1)
        circuit.ry(np.pi / 7, 2)

        # V0(theta0, theta1): Parametrized layer 0
        circuit.rz(params[0], 0)
        circuit.rz(params[1], 1)

        # W1: non-parametrized gates
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        # V_1(theta2, theta3): Parametrized layer 1
        circuit.ry(params[2], 1)
        circuit.rx(params[3], 2)

        # W2: non-parametrized gates
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        observable = I ^ I ^ Y

        overlap = self.get_overlap(circuit, backend)

        loss = []

        def callback(x, fx, stepsize, nfevs, accepted):
            loss.append(fx)

        qnspsa = QNSPSA(overlap,
                        maxiter=300,
                        blocking=True,
                        learning_rate=1e-2,
                        perturbation=1e-2,
                        tolerance=-1,
                        callback=callback)

        initial_point = np.array([0.432, -0.123, 0.543, 0.233])
        vqe = VQE(circuit, qnspsa, initial_point, quantum_instance=backend)

        _ = vqe.compute_minimum_eigenvalue(observable)

        print(loss)

    # def test_simple(self):
    #     """ test AQGD optimizer with the parameters as single values."""
    #     q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
    #                                  seed_simulator=algorithm_globals.random_seed,
    #                                  seed_transpiler=algorithm_globals.random_seed)

    #     aqgd = AQGD(momentum=0.0)
    #     vqe = VQE(var_form=RealAmplitudes(),
    #               optimizer=aqgd,
    #               gradient=Gradient('fin_diff'),
    #               quantum_instance=q_instance)
    #     result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
    #     self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    # def test_list(self):
    #     """ test AQGD optimizer with the parameters as lists. """
    #     q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
    #                                  seed_simulator=algorithm_globals.random_seed,
    #                                  seed_transpiler=algorithm_globals.random_seed)

    #     aqgd = AQGD(maxiter=[1000, 1000, 1000], eta=[1.0, 0.5, 0.3], momentum=[0.0, 0.5, 0.75])
    #     vqe = VQE(var_form=RealAmplitudes(),
    #               optimizer=aqgd,
    #               quantum_instance=q_instance)
    #     result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
    #     self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    # def test_raises_exception(self):
    #     """ tests that AQGD raises an exception when incorrect values are passed. """
    #     self.assertRaises(AlgorithmError, AQGD, maxiter=[1000], eta=[1.0, 0.5], momentum=[0.0, 0.5])

    # def test_int_values(self):
    #     """ test AQGD with int values passed as eta and momentum. """
    #     q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
    #                                  seed_simulator=algorithm_globals.random_seed,
    #                                  seed_transpiler=algorithm_globals.random_seed)


    #     aqgd = AQGD(maxiter=1000, eta=1, momentum=0)
    #     vqe = VQE(var_form=RealAmplitudes(),
    #               optimizer=aqgd,
    #               gradient=Gradient('lin_comb'),
    #               quantum_instance=q_instance)
    #     result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
    #     self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)
if __name__ == '__main__':
    unittest.main()
