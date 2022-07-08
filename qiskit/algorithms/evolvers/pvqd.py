# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The projected Variational Quantum Dynamics Algorithm."""

from typing import Optional, Union, List, Tuple, Callable

import logging
import numpy as np

from qiskit import transpile, QiskitError
from qiskit.algorithms.optimizers import Optimizer, Minimizer
from qiskit.circuit import QuantumCircuit, ParameterVector, ParameterExpression, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase,
    CircuitSampler,
    ExpectationBase,
    ListOp,
    StateFn,
)
from qiskit.opflow.gradients.circuit_gradients import ParamShift
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.utils import QuantumInstance

from .evolution_problem import EvolutionProblem
from .evolution_result import EvolutionResult
from .real_evolver import RealEvolver

logger = logging.getLogger(__name__)


class PVQDResult(EvolutionResult):
    """The result object for the pVQD algorithm."""

    def __init__(
        self,
        evolved_state: Union[StateFn, QuantumCircuit, OperatorBase],
        # TODO: aux_ops_evaluated: Optional[ListOrDict[Tuple[complex, complex]]] = None,
        aux_ops_evaluated: Optional[List[Tuple[complex, complex]]] = None,
        times: Optional[List[float]] = None,
        parameters: Optional[List[np.ndarray]] = None,
        fidelities: Optional[List[float]] = None,
        estimated_error: Optional[float] = None,
        observables: Optional[List[List[float]]] = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state.
            aux_ops_evaluated: Optional list of observables for which expected values on an evolved
                state are calculated. These values are in fact tuples formatted as (mean, standard
                deviation).
            times: The times evaluated during the time integration.
            parameters: The parameter values at each evaluation time.
            fidelities: The fidelity of the Trotter step and variational update at each iteration.
            estimated_error: The overall estimated error evaluated as product of all fidelities.
            observables: The value of the observables evaluated at each iteration.
        """
        super().__init__(evolved_state, aux_ops_evaluated)
        self.times = times
        self.parameters = parameters
        self.fidelities = fidelities
        self.estimated_error = estimated_error
        self.observables = observables


class PVQD(RealEvolver):
    """The projected Variational Quantum Dynamics Algorithm."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        timestep: float,
        optimizer: Union[Optimizer, Minimizer],
        quantum_instance: Union[Backend, QuantumInstance],
        expectation: ExpectationBase,
        initial_guess: Optional[np.ndarray] = None,
        evolution: Optional[EvolutionSynthesis] = None,
        gradients: bool = True,
    ) -> None:
        """
        Args:
            ansatz: A parameterized circuit preparing the variational ansatz to model the
                time evolved quantum state.
            initial_parameters: The initial parameters for the ansatz.
            timestep: The time step.
            optimizer: The classical optimizers used to minimize the overlap between
                Trotterization and ansatz. Can be either a :class:`.Optimizer` or a callable
                using the :class:`.Minimizer` protocol.
            quantum_instance: The backend of quantum instance used to evaluate the circuits.
            expectation: The expectation converter to evaluate expectation values.
            initial_guess: The initial guess for the first VQE optimization. Afterwards the
                previous iteration result is used as initial guess.
            evolution: The evolution synthesis to use for the construction of the Trotter step.
                Defaults to first-order Lie-Trotter decomposition.
            gradients: If True, use the parameter shift rule to compute gradients. If False,
                the optimizer will not be passed a gradient callable.
        """
        if evolution is None:
            evolution = LieTrotter()

        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.timestep = timestep
        self.optimizer = optimizer
        self.initial_guess = initial_guess
        self.expectation = expectation
        self.evolution = evolution
        self.gradients = gradients

        self._sampler = CircuitSampler(quantum_instance)

    def step(
        self, hamiltonian: OperatorBase, theta: np.ndarray, dt: float, initial_guess: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform a single time step.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            theta: The current parameters.
            dt: The time step.
            initial_guess: The initial guess for the update to minimize the fidelity.

        Returns:
            A tuple consisting of the next parameters and the fidelity of the optimization.
        """
        # construct cost function
        loss, gradient = self.get_loss(hamiltonian, dt, theta)

        # call optimizer
        if isinstance(self.optimizer, Optimizer):
            optimizer_result = self.optimizer.minimize(loss, initial_guess, gradient)
        else:
            optimizer_result = self.optimizer(loss, initial_guess, gradient)

        fidelity = 1 - optimizer_result.fun

        return theta + optimizer_result.x, fidelity

    def get_loss(
        self, hamiltonian: OperatorBase, dt: float, current_parameters: np.ndarray
    ) -> Callable[[np.ndarray], float]:
        """Get a function to evaluate the infidelity between Trotter step and ansatz.

        Args:
            hamiltonian: The Hamiltonian under which to evolve.
            dt: The time step.
            current_parameters: The current parameters.

        Returns:
            A callable to evaluate the infidelity.
        """
        # use Trotterization to evolve the current state
        trotterized = self.ansatz.bind_parameters(current_parameters)
        trotterized.append(
            PauliEvolutionGate(hamiltonian, time=dt, synthesis=self.evolution), self.ansatz.qubits
        )

        # define the overlap of the Trotterized state and the ansatz
        x = ParameterVector("w", self.ansatz.num_parameters)
        shifted = self.ansatz.assign_parameters(current_parameters + x)
        overlap = StateFn(trotterized).adjoint() @ StateFn(shifted)

        # apply the expectation converter
        converted = self.expectation.convert(overlap)

        def evaluate_loss(
            displacement: Union[np.ndarray, List[np.ndarray]]
        ) -> Union[float, List[float]]:
            """Evaluate the overlap of the ansatz with the Trotterized evolution.

            Args:
                displacement: The parameters for the ansatz.

            Returns:
                The fidelity of the ansatz with parameters ``theta`` and the Trotterized evolution.
            """
            if isinstance(displacement, list):
                displacement = np.asarray(displacement)
                value_dict = {x_i: displacement[:, i].tolist() for i, x_i in enumerate(x)}
            else:
                value_dict = dict(zip(x, displacement))

            sampled = self._sampler.convert(converted, params=value_dict)
            return 1 - np.abs(sampled.eval()) ** 2

        if self._check_gradient_supported() and self.gradients:

            def evaluate_gradient(displacement: np.ndarray) -> np.ndarray:
                """Evaluate the gradient with the parameter-shift rule.

                This is hardcoded here since the gradient framework does not support computing
                gradients for overlaps.

                Args:
                    displacement: The parameters for the ansatz.

                Returns:
                    The gradient.
                """
                # construct lists where each element is shifted by plus (or minus) pi/2
                dim = displacement.size
                plus_shifts = (displacement + np.pi / 2 * np.identity(dim)).tolist()
                minus_shifts = (displacement - np.pi / 2 * np.identity(dim)).tolist()

                evaluated = evaluate_loss(plus_shifts + minus_shifts)

                gradient = (evaluated[:dim] - evaluated[dim:]) / 2

                return gradient

        else:
            evaluate_gradient = None

        return evaluate_loss, evaluate_gradient

    def _check_gradient_supported(self) -> bool:
        """Check whether we can apply a simple parameter shift rule to obtain gradients."""

        # check whether the circuit can be unrolled to supported gates
        try:
            unrolled = transpile(
                self.ansatz, basis_gates=ParamShift.SUPPORTED_GATES, optimization_level=0
            )
        except QiskitError:
            # failed to map to supported basis
            logger.log(
                logging.INFO,
                "No gradient support: Failed to unroll to gates supported by parameter-shift.",
            )
            return False

        # check whether all parameters are unique and we do not need to apply the chain rule
        # (since it's not implemented yet)
        all_parameters = []
        for circuit_instruction in unrolled.data:
            for param in circuit_instruction.operation.params:
                if isinstance(param, ParameterExpression):
                    if isinstance(param, Parameter):
                        all_parameters.append(param)
                    else:
                        logger.log(
                            logging.INFO,
                            "No gradient support: Circuit is only allowed to have plain parameters, "
                            "as the chain rule is not yet implemented.",
                        )
                        return False

        if len(all_parameters) != self.ansatz.num_parameters:
            logger.log(
                logging.INFO,
                "No gradient support: Circuit is only allowed to have unique parameters, "
                "as the product rule is not yet implemented.",
            )
            return False

        return True

    def _get_observable_evaluator(self, observables):
        if isinstance(observables, list):
            observables = ListOp(observables)

        expectation_value = StateFn(observables, is_measurement=True) @ StateFn(self.ansatz)
        converted = self.expectation.convert(expectation_value)

        ansatz_parameters = self.ansatz.parameters

        def evaluate_observables(theta: np.ndarray) -> Union[float, List[float]]:
            """Evaluate the observables for the ansatz parameters ``theta``.

            Args:
                theta: The ansatz parameters.

            Returns:
                The observables evaluated at the ansatz parameters.
            """
            value_dict = dict(zip(ansatz_parameters, theta))
            sampled = self._sampler.convert(converted, params=value_dict)
            return sampled.eval()

        return evaluate_observables

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Args:
            evolution_problem: The evolution problem containing the hamiltonian, total evolution
                time and observables to evaluate.

        Returns:
            A result object containing the evolution information and evaluated observables.

        Raises:
            ValueError: If the evolution time is not positive or the timestep is too small.
        """
        time = evolution_problem.time
        observables = evolution_problem.aux_operators
        hamiltonian = evolution_problem.hamiltonian

        if not 0 < self.timestep <= time:
            raise ValueError(
                f"The time step ({self.timestep}) must be larger than 0 and smaller equal "
                f"the evolution time ({time})."
            )

        # get the function to evaluate the observables for a given set of ansatz parameters
        if observables is not None:
            evaluate_observables = self._get_observable_evaluator(observables)
            observable_values = [evaluate_observables(self.initial_parameters)]

        fidelities = [1]
        times = [0]
        parameters = [self.initial_parameters]

        current_time = 0

        if self.initial_guess is None:
            initial_guess = np.random.random(self.initial_parameters.size) * 0.01
        else:
            initial_guess = self.initial_guess

        while current_time < time:
            # perform VQE to find the next parameters
            next_parameters, fidelity = self.step(
                hamiltonian, parameters[-1], self.timestep, initial_guess
            )

            # set initial guess to last parameter update
            initial_guess = next_parameters - parameters[-1]

            # store parameters
            parameters.append(next_parameters)
            fidelities.append(fidelity)
            if observables is not None:
                observable_values.append(evaluate_observables(next_parameters))

            # increase time
            current_time += self.timestep
            times.append(current_time)

        evolved_state = self.ansatz.bind_parameters(parameters[-1])

        result = PVQDResult(
            evolved_state=evolved_state,
            times=times,
            parameters=parameters,
            fidelities=fidelities,
            estimated_error=np.prod(fidelities),
        )
        if observables is not None:
            result.observables = observable_values
            result.aux_ops_evaluated = observable_values[-1]

        return result
