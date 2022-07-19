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

"""Variational Quantum Real Time Evolution algorithm."""
from typing import Optional, Union, Type, Callable

import numpy as np
from scipy.integrate import OdeSolver

from qiskit.algorithms.evolvers.real_evolver import RealEvolver
from qiskit.opflow import ExpectationBase
from qiskit.utils import QuantumInstance
from .solvers.ode.forward_euler_solver import ForwardEulerSolver
from .variational_principles import RealVariationalPrinciple
from .var_qte import VarQTE


class VarQRTE(VarQTE, RealEvolver):
    """Variational Quantum Real Time Evolution algorithm.

    .. code-block::python

        from qiskit.algorithms import EvolutionProblem
        from qiskit.algorithms import VarQITE
        from qiskit import BasicAer
        from qiskit.circuit.library import EfficientSU2
        from qiskit.opflow import SummedOp, I, Z, Y, X
        from qiskit.algorithms.evolvers.variational.variational_principles.
        real_mc_lachlan_principle import (
            RealMcLachlanPrinciple,
        )
        from qiskit.algorithms import EvolutionProblem
        import numpy as np

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
        parameters = ansatz.parameters
        init_param_values = np.zeros(len(ansatz.parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        param_dict = dict(zip(parameters, init_param_values))
        var_principle = RealMcLachlanPrinciple()
        backend = BasicAer.get_backend("statevector_simulator")
        time = 1
        evolution_problem = EvolutionProblem(observable, time, ansatz, param_value_dict=param_dict)
        var_qrte = VarQRTE(var_principle, ode_function, quantum_instance=backend)
        evolution_result = var_qite.evolve(evolution_problem)
    """

    def __init__(
        self,
        variational_principle: RealVariationalPrinciple,
        ode_solver: Union[Type[OdeSolver], str] = ForwardEulerSolver,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        ode_num_t_steps: int = 15,
        expectation: Optional[ExpectationBase] = None,
        imag_part_tol: float = 1e-7,
        num_instability_tol: float = 1e-7,
        is_error_based: bool = False,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            lse_solver: Linear system of equations solver callable. It accepts ``A`` and ``b`` to
                solve ``Ax=b`` and returns ``x``. If ``None``, the default ``np.linalg.lstsq``
                solver is used.
            ode_num_t_steps: Number of ODE steps. Only relevant in case of the
                ``ForwardEulerSolver``.
            expectation: An instance of ``ExpectationBase`` which defines a method for calculating
                a metric tensor and an evolution gradient and, if provided, expectation values of
                ``EvolutionProblem.aux_operators``.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            num_instability_tol: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be
                non-negative.
            is_error_based: If ``True``, uses the argument that minimizes error bounds when solving
                differential equations. Currently not supported and switches to ``False``
                automatically.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
        """
        super().__init__(
            variational_principle,
            ode_solver,
            lse_solver=lse_solver,
            ode_num_t_steps=ode_num_t_steps,
            expectation=expectation,
            imag_part_tol=imag_part_tol,
            num_instability_tol=num_instability_tol,
            is_error_based=is_error_based,
            quantum_instance=quantum_instance,
        )
