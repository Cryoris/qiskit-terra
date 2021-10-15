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
from typing import Union, Dict

from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    metric_tensor_calculator,
    evolution_grad_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitQFI, CircuitGradient, OperatorBase


class ImaginaryMcLachlanVariationalPrinciple(ImaginaryVariationalPrinciple):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
    ):
        super().__init__(
            qfi_method,
            grad_method,
        )

    def _get_raw_metric_tensor(
        self,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        raw_metric_tensor_real = metric_tensor_calculator.calculate(
            ansatz, list(param_dict.keys()), self._qfi_method
        )

        return raw_metric_tensor_real

    def _get_raw_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        raw_evolution_grad_real = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, list(param_dict.keys()), self._grad_method
        )

        return raw_evolution_grad_real

    @staticmethod
    def _calc_metric_tensor(
        raw_metric_tensor: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return raw_metric_tensor.bind_parameters(param_dict) / 4.0

    @staticmethod
    def _calc_evolution_grad(
        raw_evolution_grad: OperatorBase, param_dict: Dict[Parameter, Union[float, complex]]
    ) -> OperatorBase:
        return -raw_evolution_grad.bind_parameters(param_dict)
