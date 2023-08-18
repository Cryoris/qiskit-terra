# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of the abstract class UCPauliRotGate for uniformly controlled
(also called multiplexed) single-qubit rotations around the Z-axes
(i.e., uniformly controlled R_z rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_z(a_i) is applied to the target qubit.
"""
import math
from typing import List, Sequence

from qiskit.circuit.quantumcircuit import QuantumCircuit, QubitSpecifier
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate as NewUCRZGate
from qiskit.utils.deprecation import deprecate_func


class UCRZGate(NewUCRZGate):
    """
    Uniformly controlled rotations (also called multiplexed rotations).
    The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al.
    (https://arxiv.org/pdf/quant-ph/0406176.pdf)
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.UCRZGate."
    )
    def __init__(self, angle_list):
        super().__init__(angle_list)


@deprecate_func(
    since="0.45.0",
    additional_msg="Instead, append a qiskit.circuit.library.UCRZGate to the circuit.",
)
def ucrz(
    self, angle_list: List[float], q_controls: Sequence[QubitSpecifier], q_target: QubitSpecifier
):
    r"""Attach a uniformly controlled (also called multiplexed gates) Rz rotation gate to a circuit.

    The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.

    Args:
        angle_list (List[float]): list of (real) rotation angles :math:`[a_0,...,a_{2^k-1}]`
        q_controls (Sequence[QubitSpecifier]): list of k control qubits
            (or empty list if no controls). The control qubits are ordered according to their
            significance in increasing order: For example if ``q_controls=[q[0],q[1]]``
            (with ``q = QuantumRegister(2)``), the rotation ``Rx(a_0)`` is performed if ``q[0]``
            and ``q[1]`` are in the state zero, the rotation ``Rx(a_1)`` is performed if ``q[0]``
            is in the state one and ``q[1]`` is in the state zero, and so on
        q_target (QubitSpecifier): target qubit, where we act on with
            the single-qubit rotation gates

    Returns:
        QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.

    Raises:
        QiskitError: if the list number of control qubits does not correspond to the provided
            number of single-qubit unitaries; if an input is of the wrong type
    """

    if isinstance(q_controls, QuantumRegister):
        q_controls = q_controls[:]
    if isinstance(q_target, QuantumRegister):
        q_target = q_target[:]
        if len(q_target) == 1:
            q_target = q_target[0]
        else:
            raise QiskitError(
                "The target qubit is a QuantumRegister containing more than one qubit."
            )
    # Check if q_controls has type "list"
    if not isinstance(angle_list, list):
        raise QiskitError("The angles must be provided as a list.")
    num_contr = math.log2(len(angle_list))
    if num_contr < 0 or not num_contr.is_integer():
        raise QiskitError(
            "The number of controlled rotation gates is not a non-negative power of 2."
        )
    # Check if number of control qubits does correspond to the number of rotations
    if num_contr != len(q_controls):
        raise QiskitError(
            "Number of controlled rotations does not correspond to the number of control-qubits."
        )

    print([q_target] + q_controls)
    return self.append(UCRZGate(angle_list), [q_target] + q_controls, [])


QuantumCircuit.ucrz = ucrz
