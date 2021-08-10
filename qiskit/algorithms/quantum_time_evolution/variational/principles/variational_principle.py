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
from abc import ABC, abstractmethod
from typing import Union

from qiskit.algorithms.quantum_time_evolution.variational.principles import metric_tensor_calculator, \
    evolution_grad_calculator
from qiskit.opflow import CircuitQFI, CircuitGradient


class VariationalPrinciple(ABC):
    def __init__(self, observable, ansatz, parameters,
                 qfi_method: Union[str, CircuitQFI] = 'lin_comb_full',
                 grad_method: Union[str, CircuitGradient] = 'lin_comb',
                 is_error_supported: bool = False):
        self._is_error_supported = is_error_supported
        raw_metric_tensor = metric_tensor_calculator.build(observable, ansatz, parameters, qfi_method)
        raw_evolution_grad = evolution_grad_calculator.build(observable, ansatz, parameters,
                                                             grad_method)
        self._metric_tensor = self._calc_metric_tensor(raw_metric_tensor)
        self._evolution_grad = self._calc_evolution_grad(raw_evolution_grad)

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(raw_evolution_grad):
        pass

    @property
    def metric_tensor(self):
        return self._metric_tensor

    @property
    def evolution_grad(self):
        return self._evolution_grad
