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

"""Variational Quantum Imaginary Time Evolution algorithm."""

from typing import Optional, Union, Callable

import numpy as np
from scipy.integrate import OdeSolver, RK45

from qiskit.algorithms import ImaginaryEvolver, EvolutionProblem, eval_observables
from qiskit.algorithms.evolvers.evolution_result import EvolutionResult
from qiskit.algorithms.evolvers.variational.variational_principles.imaginary.imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.algorithms.evolvers.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)
from qiskit.opflow import (
    StateFn,
    ExpectationBase,
)
from qiskit.algorithms.evolvers.variational.algorithms.var_qte import VarQTE
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQITE(ImaginaryEvolver, VarQTE):
    """Variational Quantum Imaginary Time Evolution algorithm."""

    def __init__(
        self,
        variational_principle: ImaginaryVariationalPrinciple,
        ode_function_generator: AbstractOdeFunctionGenerator,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        expectation: Optional[ExpectationBase] = None,
        ode_solver_callable: OdeSolver = RK45,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        allowed_imaginary_part: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
    ) -> None:
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            ode_function_generator: Generator for a function that ODE will use.
            backend: Backend used to evaluate the quantum circuit outputs.
            expectation: An instance of ExpectationBase which defines a method for calculating
                expectation values of EvolutionProblem.aux_operators.
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
            lse_solver_callable: Linear system of equations solver that follows a NumPy
                np.linalg.lstsq interface.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                rounded up to 0 for quantities that are expected to be non-negative.
        """
        super().__init__(
            variational_principle,
            ode_function_generator,
            backend,
            expectation,
            ode_solver_callable,
            lse_solver_callable,
            allowed_imaginary_part,
            allowed_num_instability_error,
        )

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given
        operator.

        Args:
            evolution_problem: Instance defining evolution problem.

        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective time evolution.
        """
        init_state_param_dict = self._create_init_state_param_dict(
            evolution_problem.hamiltonian_value_dict,
            list(evolution_problem.initial_state.parameters),
        )
        self.bind_initial_state(StateFn(evolution_problem.initial_state), init_state_param_dict)

        error_calculator = None  # TODO will be supported in another PR

        evolved_state = super()._evolve_helper(
            init_state_param_dict,
            evolution_problem.hamiltonian,
            evolution_problem.time,
            evolution_problem.t_param,
            error_calculator,
            evolution_problem.initial_state,
        )

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = eval_observables(
                self._backend, evolved_state, evolution_problem.aux_operators, self._expectation
            )

        return EvolutionResult(evolved_state, evaluated_aux_ops)
