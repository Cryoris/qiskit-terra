# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Variational Quantum Imaginary Time Evolution algorithm."""

import unittest

from qiskit.algorithms import EvolutionProblem
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.evolvers.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit import Aer
from qiskit.algorithms.evolvers.variational.variational_principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.evolvers.variational.algorithms.var_qite import VarQITE
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
)
from qiskit.quantum_info import state_fidelity, Statevector


class TestVarQITE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Imaginary Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        np.random.seed(11)

    def test_run_d_1(self):
        """Test VarQITE for d = 1 and t = 1."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))
        backend = Aer.get_backend("statevector_simulator")

        ode_function = OdeFunctionGenerator()
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        time = 1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, hamiltonian_value_dict=param_dict
        )

        evolution_result = var_qite.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        # values from the prototype
        thetas_expected = [
            0.905901128153194,
            2.00336012271211,
            2.69398302961789,
            2.74463415808318,
            2.33136908953802,
            1.7469421463254,
            2.10770301585949,
            1.92775891861825,
        ]
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)

    def test_run_d_1_t_7(self):
        """Test VarQITE for d = 1 and t = 7."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = Aer.get_backend("statevector_simulator")

        ode_function = OdeFunctionGenerator()
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        time = 7

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, hamiltonian_value_dict=param_dict
        )

        evolution_result = var_qite.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        # values from the prototype
        thetas_expected = [
            0.828917365718767,
            1.88481074798033,
            3.14111335991238,
            3.14125849601269,
            2.33768562678401,
            1.78670990729437,
            2.04214275514208,
            2.04009918594422,
        ]
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=2)

    def test_run_d_2(self):
        """Test VarQITE for d = 2 and t = 1."""
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        backend = Aer.get_backend("statevector_simulator")

        ode_function = OdeFunctionGenerator()
        var_qite = VarQITE(var_principle, ode_function, quantum_instance=backend)
        time = 1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, hamiltonian_value_dict=param_dict
        )

        evolution_result = var_qite.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        # values from the prototype
        thetas_expected = [
            1.29495364023786,
            1.08970061333559,
            0.667488228710748,
            0.500122687902944,
            1.4377736672043,
            1.22881086103085,
            0.729773048146251,
            1.01698854755226,
            0.050807780587492,
            0.294828474947149,
            0.839305697704923,
            0.663689581255428,
        ]
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
