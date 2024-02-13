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

"""Dynamical Decoupling insertion pass on multiple qubits."""

import itertools
import warnings

import numpy as np
import rustworkx as rx

from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, RZGate, CXGate, ECRGate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class DynamicalDecouplingMulti(TransformationPass):
    """Dynamical decoupling insertion pass on multi-qubit delays."""

    def __init__(self, target, skip_reset_qubits=True, pulse_alignment=1, skip_threshold=1):
        """Dynamical decoupling initializer.
        Args:
            target (Target): Target specifying gate durations and connectivity of qubits
                which will influence the DD sequences inserted.
            skip_reset_qubits (bool): if True, does not insert DD on idle
                periods that immediately follow initialized/reset qubits (as
                qubits in the ground state are less susceptile to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided om ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            skip_threshold (float): a number in range [0, 1]. If the DD sequence
                amounts to more than this fraction of the idle window, we skip.
                Default: 1 (i.e. always insert, even if filling up the window).
        """
        super().__init__()
        self._target = target
        self._durations = target.durations()
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._skip_threshold = skip_threshold

    def run(self, dag):
        """Run the DynamicalDecoupling pass on dag.
        Args:
            dag (DAGCircuit): a scheduled DAG.
        Returns:
            DAGCircuit: equivalent circuit with delays interrupted by DD,
                where possible.
        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        if dag.duration is None:
            raise TranspilerError("DD runs after circuit is scheduled.")

        new_dag = dag.copy_empty_like()

        adjacency_map = self._target.build_coupling_map()
        adjacency_map.make_symmetric()  # don't care for direction
        qubit_index_map = {qubit: index for index, qubit in enumerate(new_dag.qubits)}

        def _constrained_length(values):
            return self._alignment * np.floor(values / self._alignment)

        for nd in dag.topological_op_nodes():
            if not isinstance(nd.op, Delay):
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            dag_qubits = nd.qargs
            physical_qubits = [qubit_index_map[q] for q in dag_qubits]

            if self._skip_reset_qubits:  # discount initial delays
                pred = next(dag.predecessors(nd))
                if isinstance(pred, DAGInNode) or isinstance(pred.op, Reset):
                    for q in dag_qubits:
                        new_dag.apply_operation_back(Delay(nd.op.duration), [q], [])
                    continue

            start_time = self.property_set["node_start_time"][nd]
            end_time = start_time + nd.op.duration

            # 1) color each qubit in this delay instruction
            neighborhood = set(
                [neighbor for q in physical_qubits for neighbor in adjacency_map.neighbors(q)]
            )
            neighborhood |= set(physical_qubits)
            coloring = {i: None for i in neighborhood}
            # first color qubits that are ctrl/tgt of CX/ECR, in this neighborhood & this time interval
            for q in neighborhood:
                dag_q = dag.qubits[q]
                for q_node in dag.nodes_on_wire(dag_q, only_ops=True):
                    adj_start_time = self.property_set["node_start_time"][q_node]
                    adj_end_time = adj_start_time + q_node.op.duration
                    if (
                        adj_start_time < end_time
                        and adj_end_time > start_time
                        and isinstance(q_node.op, (CXGate, ECRGate))
                    ):
                        ctrl, tgt = [qubit_index_map[q] for q in q_node.qargs]
                        if q == ctrl:
                            coloring[q] = 0
                        if q == tgt:
                            coloring[q] = 1
            # now color delay qubits, subject to previous colors and keeping to as few colors as possible
            for physical_qubit, dag_qubit in zip(physical_qubits, dag_qubits):
                if coloring[physical_qubit] is None:
                    adjacent_colors = set(
                        coloring[neighbor]
                        for neighbor in adjacency_map.neighbors(physical_qubit)
                        if coloring[neighbor] is not None
                    )
                    color = 0
                    while color in adjacent_colors:
                        color += 1
                    coloring[physical_qubit] = color

            # 2) insert the actual DD sequences
            for physical_qubit, dag_qubit in zip(physical_qubits, dag_qubits):
                color = coloring[physical_qubit]

                if color == 0:
                    dd_sequence = [XGate(), XGate()]
                    num_pulses = 2
                    spacing = [1 / 2, 1 / 2, 0]
                elif color == 1:
                    dd_sequence = [XGate(), XGate()]
                    num_pulses = 2
                    spacing = [1 / 4, 1 / 2, 1 / 4]
                elif color == 2:
                    dd_sequence = [XGate(), XGate(), XGate(), XGate()]
                    num_pulses = 4
                    spacing = [1 / 4, 1 / 4, 1 / 4, 1 / 4, 0]
                else:
                    continue

                dd_sequence_duration = 0
                num_pulses = len(dd_sequence)
                for gate in dd_sequence:
                    dd_sequence_duration += self._durations.get(gate, physical_qubit)
                slack = nd.op.duration - dd_sequence_duration
                slack_fraction = slack / nd.op.duration
                if 1 - slack_fraction >= self._skip_threshold:  # dd doesn't fit
                    new_dag.apply_operation_back(Delay(nd.op.duration), [dag_qubit], [])
                    continue

                taus = _constrained_length(slack * np.asarray(spacing))
                unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
                middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
                to_middle = _constrained_length(unused_slack)
                taus[middle_index] += to_middle  # now we add up to original delay duration
                if unused_slack - to_middle:
                    taus[-1] += unused_slack - to_middle

                sequence_gphase = 0
                if num_pulses != 1:
                    if num_pulses % 2 != 0:
                        raise TranspilerError(
                            "DD sequence must contain an even number of gates (or 1)."
                        )
                    noop = np.eye(2)
                    for gate in dd_sequence:
                        noop = noop.dot(gate.to_matrix())
                    if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                        raise TranspilerError(
                            "The DD sequence does not make an identity operation."
                        )
                    sequence_gphase = np.angle(noop[0][0])

                for tau, gate in itertools.zip_longest(taus, dd_sequence):
                    if tau > 0:
                        new_dag.apply_operation_back(Delay(tau), [dag_qubit])
                    if gate is not None:
                        new_dag.apply_operation_back(gate, [dag_qubit])
                new_dag.global_phase = _mod_2pi(new_dag.global_phase + sequence_gphase)

        return new_dag


def _mod_2pi(angle: float, atol: float = 0):
    """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped
