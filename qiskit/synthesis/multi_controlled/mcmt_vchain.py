# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesis for multiple-control, multiple-target Gate."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Gate


def synth_mcmt_vchain(gate: Gate, num_ctrl_qubits: int, num_target_qubits: int) -> QuantumCircuit:
    """Synthesize MCMT using a V-chain.

    This uses a chain of CCX gates, using ``num_ctrl_qubits - 1`` auxiliary qubits.

    For example, a 3-control and 2-target H gate will be synthesized as::

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘

    """
    num_qubits = 2 * num_ctrl_qubits - 1 + num_target_qubits
    circuit = QuantumCircuit(num_qubits)

    control_qubits = list(range(num_ctrl_qubits))
    target_qubits = list(range(num_ctrl_qubits, num_ctrl_qubits + num_target_qubits))
    ancilla_qubits = list(range(num_ctrl_qubits + num_target_qubits, num_qubits))

    if len(ancilla_qubits) > 0:
        master_control = ancilla_qubits[-1]
    else:
        master_control = control_qubits[0]

    controlled_gate = gate.control()
    _apply_v_chain(circuit, control_qubits, ancilla_qubits, reverse=False)
    for qubit in target_qubits:
        circuit.append(controlled_gate, [master_control, qubit], [])
    _apply_v_chain(circuit, control_qubits, ancilla_qubits, reverse=True)

    return circuit


def _apply_v_chain(
    circuit: QuantumCircuit,
    control_qubits: list[int],
    ancilla_qubits: list[int],
    reverse: bool = False,
) -> None:
    """Get the rule for the CCX V-chain.

    The CCX V-chain progressively computes the CCX of the control qubits and puts the final
    result in the last ancillary qubit.

    Args:
        control_qubits: The control qubits.
        ancilla_qubits: The ancilla qubits.
        reverse: If True, compute the chain down to the qubit. If False, compute upwards.

    Returns:
        The rule for the (reversed) CCX V-chain.
    """
    iterations = list(enumerate(range(2, len(control_qubits))))
    if not reverse:
        circuit.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
        for i, j in iterations:
            circuit.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
    else:
        for i, j in reversed(iterations):
            circuit.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
        circuit.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
