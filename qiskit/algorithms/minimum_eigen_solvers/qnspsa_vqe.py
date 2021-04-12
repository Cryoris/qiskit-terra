# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List, Callable, Union
import logging
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow import OperatorBase, ExpectationBase, I, CircuitSampler
from qiskit.utils.quantum_instance import QuantumInstance

from qiskit.algorithms.optimizers import SPSA, QNSPSA

from .minimum_eigen_solver import MinimumEigensolverResult
from .vqe import VQE, VQEResult
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)

# disable check for ansatzs, optimizer setter because of pylint bug
# pylint: disable=no-member


class QNSPSAVQE(VQE):
    r"""The Variational Quantum Eigensolver algorithm.

    `VQE <https://arxiv.org/abs/1304.3061>`__ is a hybrid algorithm that uses a
    variational technique and interleaves quantum and classical computations in order to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.

    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (a.k.a. ansatz) which is a :class:`QuantumCircuit`, and one of the classical
    :mod:`~qiskit.algorithms.optimizers`. The ansatz is varied, via its set of parameters, by the
    optimizer, such that it works towards a state, as determined by the parameters applied to the
    variational form, that will result in the minimum expectation value being measured of the input
    operator (Hamiltonian).

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  It provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the variational form being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the variational form for a preferred value, based on its
    given initial state. If the variational form returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the variational form provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the variational form returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    .. note::

        The VQE stores the parameters of ``ansatz`` sorted by name to map the values
        provided by the optimizer to the circuit. This is done to ensure reproducible results,
        for example such that running the optimization twice with same random seeds yields the
        same result. Also, the ``optimal_point`` of the result object can be used as initial
        point of another VQE run by passing it as ``initial_point`` to the initializer.

    """

    def __init__(self,
                 ansatz: Optional[QuantumCircuit] = None,
                 initial_point: Optional[np.ndarray] = None,
                 expectation: Optional[ExpectationBase] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                 natural_spsa: bool = False,
                 maxiter: int = 100,
                 blocking: bool = True,
                 allowed_increase: float = 0.1,
                 learning_rate: float = 0.01,
                 perturbation: float = 0.01,
                 regularization: float = 0.01,
                 resamplings: int = 1,
                 hessian_delay: int = 0,
                 ) -> None:
        """

        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.`
            quantum_instance: Quantum Instance or Backend
        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        # set the initial point to the preferred parameters of the variational form
        if initial_point is None and hasattr(ansatz, 'preferred_init_points'):
            initial_point = ansatz.preferred_init_points

        self._circuit_sampler = None  # type: Optional[CircuitSampler]
        self._expectation = expectation
        self._user_valid_expectation = self._expectation is not None
        self._expect_op = None

        super().__init__(ansatz=ansatz,
                         initial_point=initial_point,
                         quantum_instance=quantum_instance)

        self.natural_spsa = natural_spsa
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.allowed_increase = allowed_increase
        self.blocking = blocking
        self.regularization = regularization
        self.resamplings = resamplings
        self.hessian_delay = hessian_delay

        self._ret = VQEResult()
        self._eval_time = None
        self._callback = callback

        self._eval_count = 0
        logger.info(self.print_settings())

    @property
    def optimizer(self):  # pylint: disable=arguments-differ
        raise NotImplementedError('The optimizer is a SPSA version with batched circuits and '
                                  'cannot be returned as a standalone.')

    @optimizer.setter
    def optimizer(self, optimizer):
        raise NotImplementedError('The optimizer is a SPSA version with batched circuits and '
                                  'cannot be set.')

    def compute_minimum_eigenvalue(
            self,
            operator: OperatorBase,
            aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        if self.quantum_instance is None:
            raise AlgorithmError("A QuantumInstance or Backend "
                                 "must be supplied to run the quantum algorithm.")

        if operator is None:
            raise AlgorithmError("The operator was never provided.")

        operator = self._check_operator(operator)
        # We need to handle the array entries being Optional i.e. having value None
        if aux_operators:
            zero_op = I.tensorpower(operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]
        else:
            aux_operators = None

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0

        if not self._expect_op:
            self._expect_op = self.construct_expectation(self._ansatz_params, operator)

        # CODE BELOW
        optimizer_settings = {'maxiter': self.maxiter,
                              'blocking': self.blocking,
                              'allowed_increase': self.allowed_increase,
                              'learning_rate': self.learning_rate,
                              'perturbation': self.perturbation,
                              'regularization': self.regularization,
                              'resamplings': self.resamplings,
                              'hessian_delay': self.hessian_delay,
                              'callback': self._callback,
                              'backend': self._quantum_instance}

        if self.natural_spsa:
            optimizer = QNSPSA(overlap_fn=self.ansatz, **optimizer_settings)
        else:
            optimizer = SPSA(**optimizer_settings)

        vqresult = self.find_minimum(initial_point=self.initial_point,
                                     ansatz=self.ansatz,
                                     cost_fn=self._expect_op,
                                     optimizer=optimizer)

        self._ret = VQEResult()
        self._ret.combine(vqresult)

        if vqresult.optimizer_evals is not None and \
                self._eval_count >= vqresult.optimizer_evals:
            self._eval_count = vqresult.optimizer_evals
        self._eval_time = vqresult.optimizer_time
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, vqresult.optimal_point, self._eval_count)

        self._ret.eigenvalue = vqresult.optimal_value + 0j
        self._ret.eigenstate = self.get_optimal_vector()
        self._ret.eigenvalue = self.get_optimal_cost()
        if aux_operators:
            self._eval_aux_ops(aux_operators)
            self._ret.aux_operator_eigenvalues = self._ret.aux_operator_eigenvalues[0]

        self._ret.cost_function_evals = self._eval_count

        return self._ret
