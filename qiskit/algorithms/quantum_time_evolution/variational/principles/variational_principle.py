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
from typing import Union, Dict, Optional

from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    OperatorBase,
    StateFn,
    NaturalGradient,
    PauliExpectation,
)


class VariationalPrinciple(ABC):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
        is_error_supported: Optional[bool] = False,
    ):
        self._qfi_method = qfi_method
        self._grad_method = grad_method
        self._is_error_supported = is_error_supported

    def _lazy_init(
        self,
        hamiltonian,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
        regularization: str,
    ):
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._param_dict = param_dict
        self._operator = ~StateFn(hamiltonian) @ StateFn(ansatz)
        self._operator = self._operator / self._operator.coeff  # Remove the time from the operator
        raw_metric_tensor = self._get_raw_metric_tensor(ansatz, param_dict)
        print("Raw metric tensor")
        print(raw_metric_tensor.assign_parameters(param_dict).to_matrix())

        raw_evolution_grad = self._get_raw_evolution_grad(hamiltonian, ansatz, param_dict)
        print("Raw evolution grad")
        print(raw_evolution_grad.assign_parameters(param_dict).to_matrix())

        self._metric_tensor = self._calc_metric_tensor(raw_metric_tensor, param_dict)
        self._evolution_grad = self._calc_evolution_grad(raw_evolution_grad, param_dict)
        self._nat_grad = self._calc_nat_grad(self._operator, param_dict, regularization)

    @abstractmethod
    def _get_raw_metric_tensor(
        self,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        pass

    @abstractmethod
    def _get_raw_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        param_dict: Dict[Parameter, Union[float, complex]],
    ):
        pass

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor, param_dict: Dict[Parameter, Union[float, complex]]):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(
        raw_evolution_grad, param_dict: Dict[Parameter, Union[float, complex]]
    ):
        pass

    @abstractmethod
    def _calc_nat_grad(
        self,
        raw_operator: OperatorBase,
        param_dict: Dict[Parameter, Union[float, complex]],
        regularization: str,
    ) -> OperatorBase:
        nat_grad = NaturalGradient(
            grad_method=self._grad_method,
            qfi_method=self._qfi_method,
            regularization=regularization,
        ).convert(raw_operator * 0.5, list(param_dict.keys()))

        # TODO should be bind here? also need to bind time as ODE progresses
        # nat_grad = nat_grad.bind_parameters(param_dict)

        return PauliExpectation().convert(nat_grad)

    @abstractmethod
    def _calc_error_bound(
        self, error: float, et: float, h_squared_expectation, h_trip: float, trained_energy: float
    ) -> float:
        pass

    @property
    def metric_tensor(self) -> OperatorBase:
        return self._metric_tensor

    @property
    def evolution_grad(self) -> OperatorBase:
        return self._evolution_grad

    @staticmethod
    def op_real_part(operator: OperatorBase) -> OperatorBase:
        return (operator + operator.adjoint()) / 2.0

    @staticmethod
    def op_imag_part(operator: OperatorBase) -> OperatorBase:
        return (operator - operator.adjoint()) / (2.0j)
