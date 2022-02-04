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

"""Abstract class for generating ODE functions."""

from abc import ABC, abstractmethod
from typing import Iterable, Union, Dict, Optional

from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class AbstractOdeFunctionGenerator(ABC):
    """Abstract class for generating ODE functions."""

    def __init__(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
        variational_principle: VariationalPrinciple,
        grad_circ_sampler: Optional[CircuitSampler] = None,
        metric_circ_sampler: Optional[CircuitSampler] = None,
        energy_sampler: Optional[CircuitSampler] = None,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        t_param: Optional[Parameter] = None,
        allowed_imaginary_part: float = 1e-7,
    ):
        """
        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            variational_principle: Variational Principle to be used.
            grad_circ_sampler: CircuitSampler for evolution gradients.
            metric_circ_sampler: CircuitSampler for metric tensors.
            energy_sampler: CircuitSampler for energy.
            regularization: Use the following regularization with a least square method to solve the
                            underlying system of linear equations.
                            Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                            ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search,
                            or a penalty term given as Callable.
                            If regularization is None but the metric is ill-conditioned or singular
                            then a least square solver is used without regularization.
            backend: Optional backend tht enables the use of circuit samplers.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                                    imaginary part is expected.
        """
        self._param_dict = param_dict
        self._variational_principle = variational_principle
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._energy_sampler = energy_sampler
        self._regularization = regularization
        self._backend = backend
        self._linear_solver = VarQteLinearSolver(
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._energy_sampler,
            self._regularization,
            allowed_imaginary_part,
        )
        self._t_param = t_param
        self._allowed_imaginary_part = allowed_imaginary_part

    @abstractmethod
    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> Iterable:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.
        Args:
            time: Current time of evolution.
            parameters_values: Current values of parameters.
        Returns:
            Tuple containing natural gradient, metric tensor and evolution gradient results
            arising from solving a system of linear equations.
        """
        pass
