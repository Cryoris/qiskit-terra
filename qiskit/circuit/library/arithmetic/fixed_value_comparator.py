# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fixed Value Comparator."""

from typing import List, Optional
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.aqua.circuits.gates import logical_or  # pylint: disable=unused-import


class FixedValueComparator(QuantumCircuit):
    r"""Fixed Value Comparator.

    Operator compares basis states \|i>_n against a classically
    given fixed value L and flips a target qubit if i >= L (or < depending on parameters):

        \|i>_n\|0> --> \|i>_n\|1> if i >= L else \|i>\|0>

    Operator is based on two's complement implementation of binary
    subtraction but only uses carry bits and no actual result bits.
    If the most significant carry bit (= results bit) is 1, the ">="
    condition is True otherwise it is False.
    """

    def __init__(self, num_state_qubits: Optional[int] = None,
                 value: Optional[int] = None,
                 geq: bool = True,
                 name: str = 'cmp') -> None:
        """Create a new fixed value comparator circuit.

        Args:
            num_state_qubits: Number of state qubits. If this is set it will determine the number
                of qubits required for the circuit.
            value: The fixed value to compare with.
            geq: Evaluate ">=" condition of "<" condition.
            name: Name of the circuit.
        """
        super().__init__(name=name)

        self._data = None
        self._value = None
        self._geq = None
        self._num_state_qubits = None

        self.value = value
        self.geq = geq
        self.num_state_qubits = num_state_qubits

    @property
    def value(self) -> int:
        """The value to compare the qubit register to.

        Returns:
            The value against which the value of the qubit register is compared.
        """
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        if value != self._value:
            self._data = None  # reset data
            self._value = value

    @property
    def geq(self) -> bool:
        """Return whether the comparator compares greater or less equal.

        Returns:
            True, if the comparator compares '>=', False if '<'.
        """
        return self._geq

    @geq.setter
    def geq(self, geq: bool) -> None:
        """Set whether the comparator compares greater or less equal.

        Args:
            geq: If True, the comparator compares '>=', if False '<'.
        """
        if geq != self._geq:
            self._data = None  # reset data
            self._geq = geq

    @property
    def num_state_qubits(self) -> int:
        """The number of qubits encoding the state for the comparison.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: Optional[int]) -> None:
        """Set the number of state qubits.

        Note that this will change the quantum registers.

        Args:
            num_state_qubits: The new number of state qubits.
        """
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._data = None  # reset data
            self._num_state_qubits = num_state_qubits

            if num_state_qubits:
                # set the new qubit registers
                qr_state = QuantumRegister(self.num_state_qubits, name='state')
                q_compare = QuantumRegister(1, name='compare')

                self.qregs = [qr_state, q_compare]

                if self.num_ancilla_qubits > 0:
                    qr_ancilla = QuantumRegister(self.num_ancilla_qubits, name='ancilla')
                    self.qregs += [qr_ancilla]

    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits used.

        Returns:
            The number of ancillas in the circuit.
        """
        return self._num_state_qubits - 1

    def _get_twos_complement(self) -> List[int]:
        """Returns the 2's complement of self.value as array.

        Returns:
             The 2's complement of self.value.
        """
        twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
        twos_complement = '{0:b}'.format(twos_complement).rjust(self.num_state_qubits, '0')
        twos_complement = \
            [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
        return twos_complement

    @property
    def data(self):
        if self._data is None:
            self._build()
        return self._data

    def _configuration_is_valid(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self._num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('Number of state qubits is not set.')

        if self._value is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('No comparison value set.')

        required_num_qubits = 2 * self.num_state_qubits
        if self.num_qubits != required_num_qubits:
            valid = False
            if raise_on_failure:
                raise CircuitError('Number of qubits does not match required number of qubits.')

        return valid

    def _build(self) -> None:
        """Build the comparator circuit."""
        # set the register
        if self._data:
            return

        _ = self._configuration_is_valid()

        self._data = []

        qr_state = self.qubits[:self.num_state_qubits]
        q_compare = self.qubits[self.num_state_qubits]
        qr_ancilla = self.qubits[self.num_state_qubits + 1:]

        if self.value <= 0:  # condition always satisfied for non-positive values
            if self._geq:  # otherwise the condition is never satisfied
                self.x(q_compare)
        # condition never satisfied for values larger than or equal to 2^n
        elif self.value < pow(2, self.num_state_qubits):

            if self.num_state_qubits > 1:
                twos = self._get_twos_complement()
                for i in range(self.num_state_qubits):
                    if i == 0:
                        if twos[i] == 1:
                            self.cx(qr_state[i], qr_ancilla[i])
                    elif i < self.num_state_qubits - 1:
                        if twos[i] == 1:
                            self.OR([qr_state[i], qr_ancilla[i - 1]], qr_ancilla[i], None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
                    else:
                        if twos[i] == 1:
                            # OR needs the result argument as qubit not register, thus
                            # access the index [0]
                            self.OR([qr_state[i], qr_ancilla[i - 1]], q_compare, None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i - 1], q_compare)

                # flip result bit if geq flag is false
                if not self._geq:
                    self.x(q_compare)

                # uncompute ancillas state
                for i in reversed(range(self.num_state_qubits-1)):
                    if i == 0:
                        if twos[i] == 1:
                            self.cx(qr_state[i], qr_ancilla[i])
                    else:
                        if twos[i] == 1:
                            self.OR([qr_state[i], qr_ancilla[i - 1]], qr_ancilla[i], None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
            else:

                # num_state_qubits == 1 and value == 1:
                self.cx(qr_state[0], q_compare)

                # flip result bit if geq flag is false
                if not self._geq:
                    self.x(q_compare)

        else:
            if not self._geq:  # otherwise the condition is never satisfied
                self.x(q_compare)
