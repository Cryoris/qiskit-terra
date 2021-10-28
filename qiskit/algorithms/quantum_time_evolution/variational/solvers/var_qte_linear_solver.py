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
from typing import Union, List, Dict, Optional

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQteLinearSolver:
    def __init__(
        self,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        nat_grad_circ_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):

        self._backend = backend
        self._regularization = regularization
        if backend is not None:
            self._grad_circ_sampler = grad_circ_sampler
            self._metric_circ_sampler = metric_circ_sampler
            self._nat_grad_circ_sampler = nat_grad_circ_sampler

    def _solve_sle(
        self,
        var_principle: VariationalPrinciple,
        param_dict: Dict[Parameter, Union[float, complex]],
        t_param: Optional[Parameter] = None,
        t: Optional[float] = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.
        Args:
            var_principle: Variational Principle to be used.
            param_dict: Dictionary which relates parameter values to the parameters in the Ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            t: Time value that will be bound to t_param.
        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
                Fubini-Study Metric.
        """

        nat_grad_result = self._calc_nat_grad_result(param_dict, var_principle)
        if t_param is not None:
            time_dict = {t_param: t}
            nat_grad_result = nat_grad_result.bind_parameters(time_dict)

        return np.real(nat_grad_result)

    def _solve_sle_for_error_bounds(
        self,
        var_principle: VariationalPrinciple,
        param_dict: Dict[Parameter, Union[float, complex]],
        t_param: Parameter = None,
        t: float = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation with error bounds.
        Args:
            var_principle: Variational Principle to be used.
            param_dict: Dictionary which relates parameter values to the parameters in the Ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            t: Time value that will be bound to t_param.
        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
        Fubini-Study Metric.
        """
        metric_tensor = var_principle.metric_tensor
        evolution_grad = var_principle.evolution_grad

        # bind time parameter for the current value of time from the ODE solver

        if t_param is not None:
            time_dict = {t_param: t}
            evolution_grad = evolution_grad.bind_parameters(time_dict)
        grad_res = self._eval_evolution_grad(evolution_grad, param_dict)
        metric_res = self._eval_metric_tensor(metric_tensor, param_dict)

        self._inspect_imaginary_parts(grad_res, metric_res)

        metric_res = np.real(metric_res)
        grad_res = np.real(grad_res)

        # Check if numerical instabilities lead to a metric which is not positive semidefinite
        metric_res = self._check_and_fix_metric_psd(metric_res)

        return grad_res, metric_res

    def _calc_nat_grad_result(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
        var_principle: VariationalPrinciple,
    ):

        nat_grad = var_principle._nat_grad

        if self._backend is not None:
            nat_grad_result = self._nat_grad_circ_sampler.convert(
                nat_grad, params=param_dict
            ).eval()
        else:
            nat_grad_result = nat_grad.assign_parameters(param_dict).eval()

        if any(np.abs(np.imag(nat_grad_item)) > 1e-8 for nat_grad_item in nat_grad_result):
            raise Warning("The imaginary part of the gradient are non-negligible.")

        return nat_grad_result

    def _inspect_imaginary_parts(self, grad_res, metric_res):
        if any(np.abs(np.imag(grad_res_item)) > 1e-3 for grad_res_item in grad_res):
            raise Warning("The imaginary part of the gradient are non-negligible.")
        if np.any(
            [
                [np.abs(np.imag(metric_res_item)) > 1e-3 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res
            ]
        ):
            raise Warning("The imaginary part of the metric are non-negligible.")

    def _check_and_fix_metric_psd(self, metric_res):
        while True:
            w, v = np.linalg.eigh(metric_res)

            if not all(ew >= -1e-2 for ew in w):
                raise Warning(
                    "The underlying metric has ein Eigenvalue < ",
                    -1e-2,
                    ". Please use a regularized least-square solver for this problem.",
                )
            if not all(ew >= 0 for ew in w):
                # If not all eigenvalues are non-negative, set them to a small positive
                # value
                w = [max(1e-10, ew) for ew in w]
                # Recompose the adapted eigenvalues with the eigenvectors to get a new metric
                metric_res = np.real(v @ np.diag(w) @ np.linalg.inv(v))
            else:
                # If all eigenvalues are non-negative use the metric
                break
        return metric_res

    def _eval_metric_tensor(
        self, metric_tensor, param_dict: Dict[Parameter, Union[float, complex]]
    ):
        if self._backend is not None:
            # Get the QFI/4
            metric_res = np.array(
                self._metric_circ_sampler.convert(metric_tensor, params=param_dict).eval()
            )
        else:
            # Get the QFI/4
            metric_res = np.array(metric_tensor.assign_parameters(param_dict).eval())
        return metric_res

    def _eval_evolution_grad(
        self, evolution_grad, param_dict: Dict[Parameter, Union[float, complex]]
    ):
        if self._backend is not None:
            grad_res = np.array(
                self._grad_circ_sampler.convert(evolution_grad, params=param_dict).eval()
            )
        else:
            grad_res = np.array(evolution_grad.assign_parameters(param_dict).eval())
        return grad_res
