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

"""Test ODE function generator."""

import unittest

from scipy import linalg

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit import Aer
from qiskit.algorithms.time_evolution.variational.variational_principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    CircuitSampler,
)


class TestOdeFunctionGenerator(QiskitAlgorithmsTestCase):
    """Test ODE function generator."""

    def test_var_qte_ode_function(self):
        """Test ODE function generator."""
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

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        param_dict = {param: np.pi / 4 for param in parameters}
        backend = Aer.get_backend("statevector_simulator")

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        metric_tensor = var_principle._get_metric_tensor(ansatz, parameters)
        evolution_grad = var_principle._get_evolution_grad(observable, ansatz, parameters)

        time = 2

        linear_solver_callable = np.linalg.lstsq
        linear_solver = VarQteLinearSolver(
            metric_tensor,
            evolution_grad,
            linear_solver_callable,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )


        ode_function_generator = OdeFunctionGenerator()
        ode_function_generator._lazy_init(linear_solver, error_calculator=None, t_param=None, param_dict=param_dict)

        qte_ode_function = ode_function_generator.var_qte_ode_function(time, param_dict.values())
        # TODO check if values correct
        expected_qte_ode_function = [
            0.442145,
            -0.022081,
            0.106223,
            -0.117468,
            0.251233,
            0.321256,
            -0.062728,
            -0.036209,
            -0.509219,
            -0.183459,
            -0.050739,
            -0.093163,
        ]

        np.testing.assert_array_almost_equal(expected_qte_ode_function, qte_ode_function)

    def test_var_qte_ode_function_time_param(self):
        """Test ODE function generator with time param."""
        t_param = Parameter("t")
        observable = SummedOp(
            [
                0.2252 * t_param * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        param_dict = {param: np.pi / 4 for param in parameters}
        backend = Aer.get_backend("statevector_simulator")

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        metric_tensor = var_principle._get_metric_tensor(ansatz, parameters)
        evolution_grad = var_principle._get_evolution_grad(observable, ansatz, parameters)

        time = 2

        linear_solver_callable = np.linalg.lstsq
        ode_function_generator = OdeFunctionGenerator()
        linear_solver = VarQteLinearSolver(
            metric_tensor,
            evolution_grad,
            linear_solver_callable,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )
        ode_function_generator._lazy_init(linear_solver, error_calculator=None, t_param=t_param, param_dict=param_dict)

        qte_ode_function = ode_function_generator.var_qte_ode_function(time, param_dict.values())

        # TODO verify if values correct
        expected_qte_ode_function = [
            0.442145,
            -0.022081,
            0.106223,
            -0.117468,
            0.251233,
            0.321256,
            -0.062728,
            -0.036209,
            -0.509219,
            -0.183459,
            -0.050739,
            -0.093163,
        ]

        np.testing.assert_array_almost_equal(expected_qte_ode_function, qte_ode_function, decimal=5)


if __name__ == "__main__":
    unittest.main()
