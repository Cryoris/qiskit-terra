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

"""The evolved operator ansatz."""

from typing import Optional, Union, List
import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit import (
    Parameter,
    QuantumRegister,
    QuantumCircuit,
    ParameterExpression,
    ParameterVector,
)
from .n_local.n_local import NLocal


class EvolvedOperatorAnsatz(NLocal):
    """The evolved operator ansatz."""

    def __init__(
        self,
        operators=None,
        reps: int = 1,
        evolution=None,
        insert_barriers: bool = False,
        name: str = "EvolvedOps",
        parameter_prefix: Union[str, List[str]] = "t",
        initial_state: Optional[QuantumCircuit] = None,
    ):
        """
        Args:
            operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.
                If a circuit is passed, we assume it implements an already evolved operator and thus
                the circuit is not evolved again. Can be a single operator (circuit) or a list of
                operators (and circuits).
            reps: The number of times to repeat the evolved operators.
            evolution (Optional[EvolutionBase]): An opflow converter object to construct the evolution.
                Defaults to Trotterization.
            insert_barriers: Whether to insert barriers in between each evolution.
            name: The name of the circuit.
            parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
                will be used for each parameters. Can also be a list to specify a prefix per
                operator.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        # if evolution is None:
        #     # pylint: disable=cyclic-import
        #     from qiskit.opflow import PauliTrotterEvolution
        #
        #     evolution = PauliTrotterEvolution()

        super().__init__(
            initial_state=initial_state,
            parameter_prefix=parameter_prefix,
            reps=reps,
            insert_barriers=insert_barriers,
            name=name,
        )
        self._operators = None
        if operators is not None:
            self.operators = operators

        self._evolution = evolution

        # a list of which operators are parameterized, used for internal settings
        self._ops_are_parameterized = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        if not super()._check_configuration(raise_on_failure):
            return False

        if self.operators is None:
            if raise_on_failure:
                raise ValueError("The operators are not set.")
            return False

        return True

    @property
    def num_qubits(self) -> int:
        if self.operators is None:
            return 0

        if isinstance(self.operators, list) and len(self.operators) > 0:
            return self.operators[0].num_qubits

        return self.operators.num_qubits

    @property
    def evolution(self):
        """The evolution converter used to compute the evolution.

        Returns:
            EvolutionBase: The evolution converter used to compute the evolution.
        """
        return self._evolution

    @evolution.setter
    def evolution(self, evol) -> None:
        """Sets the evolution converter used to compute the evolution.

        Args:
            evol (EvolutionBase): An opflow converter object to construct the evolution.
        """
        self._invalidate()
        self._evolution = evol

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
            list: The operators to be evolved (and circuits) contained in this ansatz.
        """
        return self._operators

    @operators.setter
    def operators(self, operators=None) -> None:
        """Set the operators to be evolved.

        operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.
            If a circuit is passed, we assume it implements an already evolved operator and thus
            the circuit is not evolved again. Can be a single operator (circuit) or a list of
            operators (and circuits).
        """
        operators = _validate_operators(operators)
        self._invalidate()
        self._operators = operators
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    # TODO: the `preferred_init_points`-implementation can (and should!) be improved!
    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            # If an initial state was set by the user, then we want to make sure that the VQE does
            # not start from a random point. Thus, we return an all-zero initial point for the
            # optimizer which is used (unless it gets overwritten by a higher-priority setting at
            # runtime of the VQE).
            # However, in order to determine the correct length, we must build the QuantumCircuit
            # first, because otherwise the operators may not be set yet.
            self._build()
            return np.zeros(self.reps * len(self.operators), dtype=float)

    def _build_gate(self):
        """ """
        return EvolvedOperatorGate(
            operators=self.operators,
            reps=self.reps,
            evolution=self.evolution,
            insert_barriers=self._insert_barriers,
            initial_state=self.initial_state,
            label=self.name,
        )

    def _build(self):
        if self._data is not None:
            return

        # need to check configuration here to ensure the operators are not None
        self._check_configuration()
        self._data = []
        num_qubits = self.num_qubits

        qr = QuantumRegister(num_qubits, "q")
        if qr.name not in [qreg.name for qreg in self.qregs]:
            # if the register already exists, probably because of a previous composition.
            # Otherwise, add it.
            self.add_register(qr)

        self._append(
            self._build_gate(),
            qargs=self.qubits,
            cargs=[],
        )


class EvolvedOperatorGate(Gate):
    """TODO"""

    def __init__(
        self,
        operators=None,
        reps: int = 1,
        evolution=None,
        insert_barriers: bool = False,
        initial_state: Optional[QuantumCircuit] = None,
        label: str = None,
    ):
        """
        Args:
            operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.
                If a circuit is passed, we assume it implements an already evolved operator and thus
                the circuit is not evolved again. Can be a single operator (circuit) or a list of
                operators (and circuits).
            reps: The number of times to repeat the evolved operators.
            evolution (Optional[EvolutionBase]): An opflow converter object to construct the evolution.
                Defaults to
                :class:`~qiskit.opflow.evolutions.pauli_trotter_evolution.PauliTrotterEvolution`.
            insert_barriers: Whether to insert barriers in between each evolution.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            label: The label for the gate.
        Raises:
            AttributeError: if the operator list is empty
        """
        if evolution is None:
            # pylint: disable=cyclic-import
            from qiskit.opflow import PauliTrotterEvolution

            evolution = PauliTrotterEvolution()

        self.operators = operators
        self._op = {}
        self.reps = reps
        self.evolution = evolution
        self.insert_barriers = insert_barriers
        self.initial_state = initial_state

        if len(operators) == 0:
            raise AttributeError("At least one operator is needed.")
        from qiskit.opflow import PauliOp

        # determine how many parameters the circuit will contain
        num_parameters = 0
        for op in operators:
            if not isinstance(op, QuantumCircuit):
                # check if the operator is just the identity, if yes, skip it
                if isinstance(op, PauliOp):
                    sig_qubits = np.logical_or(op.primitive.x, op.primitive.z)
                    if sum(sig_qubits) == 0:
                        continue
                num_parameters += 1

        # keep a list of the parameters
        self._parameters = ParameterVector("t", reps * num_parameters)

        super().__init__(
            "EvolvedOps",
            operators[0].num_qubits,
            params=list(self._parameters),
            label=label,
        )

    def _define(self):
        """TODO"""
        coeff = Parameter("c")
        circuits = []
        bind_parameter = []
        for op in self.operators:
            # if the operator is already the evolved circuit just append it
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
                bind_parameter.append(False)
            else:
                evolved_op = self.evolution.convert((coeff * op).exp_i()).reduce()
                circuit = evolved_op.to_circuit()
                # if the operator was the identity it is amounts only to a global phase and no
                # parameter is added
                bind_parameter.append(circuit.num_parameters > 0)
                circuits.append(circuit)

        evolution = QuantumCircuit(self.num_qubits, name=self.name)

        param_it = iter(self._parameters)
        first = True
        for _ in range(self.reps):
            for bind, circuit in zip(bind_parameter, circuits):
                if first:
                    first = False
                elif self.insert_barriers:
                    evolution.barrier()

                if bind:
                    param = next(param_it)
                    bound = circuit.assign_parameters({coeff: param})
                else:
                    bound = circuit

                evolution.compose(bound, inplace=True)

        if self.initial_state:
            evolution.compose(self.initial_state, front=True, inplace=True)

        # cast global phase to float if it has no free parameters
        if isinstance(evolution.global_phase, ParameterExpression):
            try:
                evolution.global_phase = float(evolution.global_phase._symbol_expr)
            # RuntimeError is raised if symengine is used, for SymPy it is a TypeError
            except (RuntimeError, TypeError):
                # expression contains free parameters
                pass
        self.definition = evolution
        coeff = Parameter("c")
        circuits = []

        for op in self.operators:
            # if the operator is already the evolved circuit just append it
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
            else:
                # check if the operator is just the identity, if yes, skip it
                if _is_pauli_identity(op):
                    continue

                evolved_op = self.evolution.convert((coeff * op).exp_i()).reduce()
                circuits.append(evolved_op.to_circuit())

        # self.rotation_blocks = []
        # self.entanglement_blocks = circuits

        super()._define()


def _validate_operators(operators):
    if not isinstance(operators, list):
        operators = [operators]

    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
            raise ValueError("All operators must act on the same number of qubits.")

    return operators


def _validate_prefix(parameter_prefix, operators):
    if isinstance(parameter_prefix, str):
        return len(operators) * [parameter_prefix]
    if len(parameter_prefix) != len(operators):
        raise ValueError("The number of parameter prefixes must match the operators.")

    return parameter_prefix


def _is_pauli_identity(operator):
    from qiskit.opflow import PauliOp

    if isinstance(operator, PauliOp):
        return not np.any(np.logical_or(operator.primitive.x, operator.primitive.z))
    return False
