# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Eigensolver algorithm, optimized for diagonal Hamiltonians."""

from __future__ import annotations

import logging
from time import time
from collections.abc import Callable

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils.validation import validate_min, validate_range
from qiskit.primitives import BaseSampler

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import SLSQP, Minimizer, Optimizer
from .sampling_mes import (
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
)
from .diagonal_estimator import diagonal_estimation

logger = logging.getLogger(__name__)


class SamplingVQE(SamplingMinimumEigensolver):
    r"""The Variational Quantum Eigensolver algorithm.

    Attributes:
        ansatz: A parameterized circuit, preparing the ansatz for the wave function. If not
            provided, this defaults to a :class:`.RealAmplitudes` circuit.
        optimizer: A classical optimizer to find the minimum energy. This can either be a
            Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer` protocol.
            Defaults to :class:`.SLSQP`.
        initial_point: An optional initial point (i.e. initial parameter values) for the optimizer.
            If not provided, a random initial point with values in the interval :math:`[0, 2\pi]`
            is used.
        sampler: The sampler primitive to sample the circuits.
        max_evals_grouped: Specifies how many parameter sets can be evaluated simultaneously.
            This information is forwarded to the optimizer, which can use it for batch evaluation.
        aggregation: A float or callable to specify how the objective function evaluated on the
            basis states should be aggregated. If a float, this specifies the :math:`\alpha \in [0,1]`
            parameter for a CVaR expectation value (see also [1]).

    References:

        [1] Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
            "Improving Variational Quantum Optimization using CVaR"
            `arXiv:1907.04769 <https://arxiv.org/abs/1907.04769>`_
    """

    def __init__(
        self,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: np.ndarray | None = None,
        sampler: BaseSampler | None = None,
        # TODO
        # gradient:
        max_evals_grouped: int = 1,
        aggregation: float | Callable[[list[float]], float] | None = None,
    ) -> None:
        """

        Args:
            ansatz: The parameterized circuit used as ansatz for the wave function.
            optimizer: The classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            sampler: The sampler primitive to sample the circuits.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.

        """
        super().__init__()

        validate_min("max_evals_grouped", max_evals_grouped, 1)
        if aggregation is not None:
            validate_range("aggregation", aggregation, 0, 1)

        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.sampler = sampler
        self.max_evals_grouped = max_evals_grouped
        self.aggregation = aggregation

    def _check_operator_ansatz(
        self, operator: BaseOperator, ansatz: QuantumCircuit
    ) -> QuantumCircuit:
        """Check that the number of qubits of operator and ansatz match."""
        if operator.num_qubits != ansatz.num_qubits:
            # try to set the number of qubits on the ansatz, if possible
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", {operator.num_qubits}
                )
                ansatz.num_qubits = operator.num_qubits
            except AttributeError as ex:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from ex

        return ansatz

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> SamplingMinimumEigensolverResult:
        # super().compute_minimum_eigenvalue(operator, aux_operators)
        # set defaults
        if self.ansatz is None:
            ansatz = RealAmplitudes(num_qubits=operator.num_qubits)
        else:
            ansatz = self.ansatz.copy()

        # check that the number of qubits of operator and ansatz match, and resize if possible
        ansatz = self._check_operator_ansatz(operator, ansatz)
        ansatz.measure_all()

        if self.sampler is None:
            raise ValueError("The sampler is None, but must be set.")

        optimizer = SLSQP() if self.optimizer is None else self.optimizer

        if isinstance(optimizer, Optimizer):
            # note that this changes the optimizer instance -- should we reset after the VQE run?
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        if self.initial_point is None:
            initial_point = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
        elif len(self.initial_point) != ansatz.num_parameters:
            raise ValueError(
                f"The dimension of the initial point ({len(self.initial_point)}) does not match the "
                f"number of parameters in the circuit ({ansatz.num_parameters})."
            )
        else:
            initial_point = self.initial_point

        # set an expectation for this algorithm run (will be reset to None at the end)
        # initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        energy_evaluation, best_measurement = self.get_energy_evaluation(
            operator, ansatz, return_best_measurement=True
        )

        start_time = time()

        if callable(optimizer):
            opt_result = optimizer(  # pylint: disable=not-callable
                fun=energy_evaluation, x0=initial_point  # , jac=gradient, bounds=bounds
            )
        else:
            opt_result = optimizer.minimize(
                fun=energy_evaluation, x0=initial_point  # , jac=gradient, bounds=bounds
            )

        eval_time = time() - start_time

        final_state = self.sampler.run([ansatz], [opt_result.x]).result().quasi_dists

        result = SamplingVQEResult()
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(ansatz.parameters, opt_result.x))
        result.cost_function_evals = opt_result.nfev
        result.optimizer_time = eval_time
        result.best_measurement = best_measurement["best"]
        result.eigenvalue = opt_result.fun
        result.eigenstate = final_state

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s.",
            eval_time,
            result.optimal_point,
        )

        if aux_operators is not None:
            result.aux_operator_values = self._eval_aux_ops(ansatz, opt_result.x, aux_operators)

        return result

    def get_energy_evaluation(
        self,
        operator: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        return_best_measurement: bool = False,
    ) -> tuple[Callable[[np.ndarray], float | list[float]], dict]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.
            ansatz: The ansatz preparing the quantum state.
            return_best_measurement: If True, a handle to a dictionary containing the best
                measurement evaluated with the cost function.


        Returns:
            Energy of the hamiltonian of each parameter, and, optionally, the expectation
            converter.

        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        if ansatz.num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        best_measurement = {"best": None}

        def energy_evaluation(parameters):
            # Create dict associating each parameter with the lists of parameterization values for it
            value, best = diagonal_estimation(self.sampler, operator, ansatz, parameters)

            # keep track of the best sample
            for best_i in best:
                if (
                    best_measurement["best"] is None
                    or best_i["value"] < best_measurement["best"]["value"]
                ):
                    best_measurement["best"] = best_i

            return value if len(value) > 1 else value[0]

        if return_best_measurement:
            return energy_evaluation, best_measurement

        return energy_evaluation

    def _eval_aux_ops(self, ansatz, parameters, aux_operators):
        # convert to list if necessary and store the keys
        if isinstance(aux_operators, dict):
            is_dict = True
            keys = list(aux_operators.keys())
            aux_operators = list(aux_operators.values())
        else:
            is_dict = False

        # evaluate all aux operators
        num = len(aux_operators)
        results = diagonal_estimation(
            self.sampler, aux_operators, num * [ansatz], num * [parameters]
        )

        # bring back into the right shape and return
        if is_dict:
            return dict(zip(keys, results))

        return results


class SamplingVQEResult(SamplingMinimumEigensolverResult):
    """VQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None
        self._optimizer_time = None

    @property
    def cost_function_evals(self) -> int | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def optimizer_time(self) -> float | None:
        """Returns time the optimization took."""
        return self._optimizer_time

    @optimizer_time.setter
    def optimizer_time(self, value: float) -> None:
        """Sets time the optimization took."""
        self._optimizer_time = value
