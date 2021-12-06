# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Class for calculating gradient errors for Variational Quantum Imaginary Time Evolution."""
from typing import Union, List, Tuple, Any, Dict, Optional

import numpy as np
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)

from qiskit.circuit import Parameter
from qiskit.opflow import OperatorBase, CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class ImaginaryErrorCalculator(ErrorCalculator):
    """Class for calculating gradient errors for Variational Quantum Imaginary Time Evolution."""

    def __init__(
        self,
        h_squared: OperatorBase,
        operator: OperatorBase,
        h_squared_sampler: CircuitSampler,
        operator_sampler: CircuitSampler,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        """
        Args:
            h_squared: Squared Hamiltonian.
            operator: Operator composed of a Hamiltonian and a quantum state.
            h_squared_sampler: CircuitSampler for a squared Hamiltonian.
            operator_sampler: CircuitSampler for an operator.
            backend: Optional backend tht enables the use of circuit samplers.
        """
        super().__init__(h_squared, operator, h_squared_sampler, operator_sampler, backend)

    def _calc_single_step_error(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
        param_dict: Dict[Parameter, Union[float, complex]],
    ) -> Tuple[int, Union[np.ndarray, complex, float], Union[Union[complex, float], Any]]:
        """
        Evaluate the l2 norm of the error for a single time step of VarQITE.
        Args:
            ng_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        """
        eps_squared = 0
        h_squared_bound = self._bind_or_sample_operator(
            self._h_squared, self._h_squared_sampler, param_dict
        )
        exp_operator_bound = self._bind_or_sample_operator(
            self._operator, self._exp_operator_sampler, param_dict
        )
        eps_squared += np.real(h_squared_bound)
        eps_squared -= np.real(exp_operator_bound ** 2)

        # ⟨dtψ(ω)|dtψ(ω)〉= dtωdtω⟨dωψ(ω)|dωψ(ω)〉
        dtdt_state = self._inner_prod(ng_res, np.dot(metric, ng_res))
        eps_squared += dtdt_state

        # 2Re⟨dtψ(ω)| H | ψ(ω)〉= 2Re dtω⟨dωψ(ω)|H | ψ(ω)〉
        regrad2 = self._inner_prod(grad_res, ng_res)
        eps_squared += regrad2

        eps_squared = self._validate_epsilon_squared(eps_squared)

        return np.real(eps_squared), dtdt_state, regrad2 * 0.5

    def _calc_single_step_error_gradient(
        self,
        ng_res: Union[List, np.ndarray],
        grad_res: Union[List, np.ndarray],
        metric: Union[List, np.ndarray],
    ) -> float:
        """
        Evaluate the gradient of the l2 norm for a single time step of VarQITE.
        Args:
            ng_res: dω/dt.
            grad_res: 2Re⟨dψ(ω)/dω|H|ψ(ω).
            metric: Fubini-Study Metric.
        Returns:
            Square root of the l2 norm of the error.
        """
        grad_eps_squared = 0
        # dω_jF_ij^Q
        grad_eps_squared += np.dot(metric, ng_res) + np.dot(
            np.diag(np.diag(metric)), np.power(ng_res, 2)
        )
        # 2Re⟨dωψ(ω)|H | ψ(ω)〉
        grad_eps_squared += grad_res
        return np.real(grad_eps_squared)
