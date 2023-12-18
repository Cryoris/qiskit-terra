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
import retworkx as rx

from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, RZGate, CXGate, ECRGate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class DynamicalDecouplingMulti(TransformationPass):
    """Dynamical decoupling insertion pass on multi-qubit delays.
    """

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

            print(f'examining delay on {physical_qubits}')

            start_time = self.property_set['node_start_time'][nd]
            end_time = start_time + nd.op.duration

            # color each qubit in this delay instruction
            coloring = {i: -1 for i in physical_qubits}
            for i, dag_qubit in enumerate(dag_qubits):
                physical_qubit = physical_qubits[i]
                ctrl_spectator = tgt_spectator = False     # is the qubit a control or target spectator
                adj_colors = []                            # is any of its adjacent qubits already colored
                for adj_physical_qubit in adjacency_map.neighbors(physical_qubit):
                    adj_dag_qubit = dag.qubits[adj_physical_qubit]
                    for adj_node in dag.nodes_on_wire(adj_dag_qubit, only_ops=True):
                        adj_start_time = self.property_set['node_start_time'][adj_node]
                        adj_end_time = adj_start_time + adj_node.op.duration
                        if adj_start_time < end_time and adj_end_time > start_time:
                            if isinstance(adj_node.op, (CXGate, ECRGate)):
                                ctrl, tgt = [qubit_index_map[q] for q in adj_node.qargs]
                                if adj_physical_qubit == ctrl:
                                    ctrl_spectator = True
                                elif adj_physical_qubit == tgt:
                                    tgt_spectator = True
                            elif isinstance(adj_node.op, Delay):
                                adj_color = coloring.get(adj_physical_qubit)
                                if adj_color:
                                    adj_colors.append(adj_color)
                print(f"delay on {physical_qubit}: ",
                      f"ctrl_spectator: {ctrl_spectator}, tgt_spectator: {tgt_spectator}, adj_colors: {adj_colors}")
                if tgt_spectator and not ctrl_spectator:
                    coloring[physical_qubit] = 0
                elif not tgt_spectator and ctrl_spectator:
                    coloring[physical_qubit] = 1
                elif tgt_spectator and ctrl_spectator:
                    coloring[physical_qubit] = 2
                print(adj_colors)
                if adj_colors and max(adj_colors) >= 0:
                    coloring[physical_qubit] = max(adj_colors) + 1

            # insert the actual DD sequence
            for i, dag_qubit in enumerate(dag_qubits):
                physical_qubit = physical_qubits[i]
                color = coloring[physical_qubit]

                print(f'inserting DD: qubit {physical_qubit}, color {color}')
                if color == 0:
                    dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
                    spacing = [1/2, 1/2, 0, 0, 0]
                    addition = [1, 1, 0, 0, 0, 0]
                if color == 1:
                    dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
                    spacing = [1/4, 1/2, 0, 0, 1/4]
                    addition = [0, 1, 0, 0, 0, 1]
                elif color == 2:
                    dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi),
                                   XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
                    spacing = [1/4, 1/4, 0, 0, 1/4, 1/4, 0, 0, 0]
                    addition = [0, 1, 0, 0, 1, 0, 0, 0, 0]
                else:
                    continue


                dd_sequence_duration = 0
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
                taus[middle_index] += unused_slack  # now we add up to original delay duration

                num_pulses = len(dd_sequence)
                sequence_gphase = 0
                if num_pulses != 1:
                    if num_pulses % 2 != 0:
                        raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
                    noop = np.eye(2)
                    for gate in dd_sequence:
                        noop = noop.dot(gate.to_matrix())
                    if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                        raise TranspilerError("The DD sequence does not make an identity operation.")
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
