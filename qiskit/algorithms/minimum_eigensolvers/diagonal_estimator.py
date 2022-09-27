# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Expectation value for a diagonal observable using a sampler primitive."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Mapping
from typing import Any

from dataclasses import dataclass

import numpy as np
from qiskit.algorithms.algorithm_job import AlgorithmJob
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.primitives import BaseSampler, BaseEstimator, EstimatorResult
from qiskit.primitives.utils import init_observable
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


@dataclass(frozen=True)
class _DiagonalEstimatorResult(EstimatorResult):
    """A result from an expectation of a diagonal observable."""

    # TODO make each measurement a dataclass rather than a dict
    best_measurements: Sequence[Mapping[str, Any]] | None = None


class _DiagonalEstimator(BaseEstimator):
    """An estimator for diagonal observables."""

    def __init__(
        self,
        sampler: BaseSampler,
        aggregation: float | Callable[[Sequence[tuple[float, float]]], float] | None = None,
        callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        **options,
    ) -> None:
        r"""Evaluate a the expectation of quantum state with respect to a diagonal operator.

        Args:
            sampler: The sampler used to evaluate the circuits.
            aggregation: The aggregation function to aggregate the measurement outcomes. If a float
                this specified the CVaR :math:`\alpha` parameter.
            callback: A callback which is given the best measurements of all circuits in each
                evaluation.
            run_options: Run options for the sampler.

        """
        super().__init__(options=options)
        self.sampler = sampler
        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[ParameterView],
        **run_options,
    ) -> AlgorithmJob:
        circuit_indices = []
        for i, circuit in enumerate(circuits):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[id(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(parameters[i])
        observable_indices = []
        for observable in observables:
            index = self._observable_ids.get(id(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable = init_observable(observable)  # convert to SparsePauliOp
                _check_observable_is_diagonal(observable)  # check it's diagonal

                # add to cache
                observable_indices.append(len(self._observables))
                self._observable_ids[id(observable)] = len(self._observables)
                self._observables.append(observable)

        job = AlgorithmJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> _DiagonalEstimatorResult:
        job = self.sampler.run(
            [self._circuits[i] for i in circuits],
            parameter_values,
            **run_options,
        )
        sampler_result = job.result()
        samples = sampler_result.quasi_dists

        # a list of dictionaries containing: {state: (measurement probability, value)}
        evaluations = [
            {
                state: (probability, _evaluate_sparsepauli(state, self._observables[i]))
                for state, probability in sampled.items()
            }
            for i, sampled in zip(observables, samples)
        ]

        results = np.array([self.aggregation(evaluated.values()) for evaluated in evaluations])

        # get the best measurements
        best_measurements = []
        num_qubits = self._circuits[0].num_qubits
        for evaluated in evaluations:
            best_result = min(evaluated.items(), key=lambda x: x[1][1])
            best_measurements.append(
                {
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }
            )

        if self.callback is not None:
            self.callback(best_measurements)

        return _DiagonalEstimatorResult(
            values=results, metadata=sampler_result.metadata, best_measurements=best_measurements
        )


def _get_cvar_aggregation(alpha):
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):

        def aggregate(measurements):
            return sum(probability * value for probability, value in measurements)

    else:

        def aggregate(measurements):
            # sort by values
            sorted_measurements = sorted(measurements, key=lambda x: x[1])

            accumulated_percent = 0  # once alpha is reached, stop
            cvar = 0
            for probability, value in sorted_measurements:
                cvar += value * max(probability, alpha - accumulated_percent)
                accumulated_percent += probability
                if accumulated_percent >= alpha:
                    break

            return cvar / alpha

    return aggregate


def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> complex:
    return sum(
        coeff * _evaluate_bitstring(state, paulistring)
        for paulistring, coeff in observable.label_iter()
    )


def _evaluate_bitstring(state: int, paulistring: str) -> float:
    """Evaluate a bitstring on a Pauli label."""
    n = len(paulistring) - 1
    return np.prod(
        [-1 if state & (1 << (n - i)) else 1 for i, pauli in enumerate(paulistring) if pauli == "Z"]
    )


def _check_observable_is_diagonal(observable: SparsePauliOp) -> bool:
    is_diagonal = not np.any(observable.paulis.x)
    if not is_diagonal:
        raise ValueError("The observable must be diagonal.")
