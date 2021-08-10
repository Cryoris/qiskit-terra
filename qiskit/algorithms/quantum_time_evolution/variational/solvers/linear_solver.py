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
from typing import Union, List, Dict

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import \
    VariationalPrinciple
from qiskit.opflow import NaturalGradient, CircuitSampler, OperatorBase


class LinearSolver():
    def __init__(self, regularization=None, backend=None):
        self._backend = backend
        self._regularization = regularization
        self._grad_circ_sampler = None
        self._metric_circ_sampler = None
        if backend is not None:
            self._grad_circ_sampler = CircuitSampler(self._backend)
            self._metric_circ_sampler = CircuitSampler(self._backend)

    # TODO better name for faster, possibly move nat_grad elsewhere
    def _solve_sle(self, var_principle: VariationalPrinciple,
                   param_dict: Dict, faster=False, nat_grad: OperatorBase = None,
                   ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle
        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the Ansatz
        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
        Fubini-Study Metric
        """
        metric_tensor = var_principle\
            .metric_tensor
        evolution_grad = var_principle.evolution_grad

        grad_res = self._eval_evolution_grad(evolution_grad, param_dict)
        metric_res = self._eval_metric_tensor(metric_tensor, param_dict)

        if any(np.abs(np.imag(grad_res_item)) > 1e-3 for grad_res_item in grad_res):
            raise Warning('The imaginary part of the gradient are non-negligible.')
        if np.any([[np.abs(np.imag(metric_res_item)) > 1e-3 for metric_res_item in metric_res_row]
                   for metric_res_row in metric_res]):
            raise Warning('The imaginary part of the metric are non-negligible.')

        metric_res = np.real(metric_res)
        grad_res = np.real(grad_res)

        # Check if numerical instabilities lead to a metric which is not positive semidefinite
        metric_res = self._check_and_fix_metric_psd(metric_res)

        if faster:
            # TODO delete this comment: grad_res already corresponds to a certain var principle
            nat_grad_result = NaturalGradient.nat_grad_combo_fn(x=[grad_res * 0.5,
                                                                   metric_res],
                                                                regularization=
                                                                self._regularization)
        elif self._backend is not None:
            nat_grad_circ_sampler = CircuitSampler(self._backend, caching='all')
            nat_grad_result = \
                nat_grad_circ_sampler.convert(nat_grad, params=param_dict).eval()
        else:
            nat_grad_result = nat_grad.assign_parameters(param_dict).eval()

        if any(np.abs(np.imag(nat_grad_item)) > 1e-8 for nat_grad_item in nat_grad_result):
            raise Warning('The imaginary part of the gradient are non-negligible.')

        print('nat grad result', nat_grad_result)

        return np.real(nat_grad_result), grad_res, metric_res

    def _check_and_fix_metric_psd(self, metric_res):
        while True:
            w, v = np.linalg.eigh(metric_res)

            if not all(ew >= -1e-2 for ew in w):
                raise Warning('The underlying metric has ein Eigenvalue < ', -1e-2,
                              '. Please use a regularized least-square solver for this problem.')
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

    def _eval_metric_tensor(self, metric_tensor, param_dict):
        if self._backend is not None:
            # Get the QFI/4
            metric_res = np.array(self._metric_circ_sampler.convert(metric_tensor,
                                                                    params=param_dict).eval()) * \
                         0.25
        else:
            # Get the QFI/4
            metric_res = np.array(metric_tensor.assign_parameters(param_dict).eval()) * 0.25
        return metric_res

    def _eval_evolution_grad(self, evolution_grad, param_dict):
        if self._backend is not None:
            grad_res = np.array(self._grad_circ_sampler.convert(evolution_grad,
                                                                params=param_dict).eval())
        else:
            grad_res = np.array(evolution_grad.assign_parameters(param_dict).eval())
        return grad_res
