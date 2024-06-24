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

import numpy as np

from qiskit.circuit.delay import Delay
from qiskit.circuit.reset import Reset
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import IGate, XGate, CXGate, ECRGate
from qiskit.dagcircuit import DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
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

        # keep track of which qubits we checked for correct X gate pulse length (which
        # need to be an integer multiple of the pulse alignment)
        checked_gate_lengths = set()

        # TODO why not change the DAG inplace?
        for nd in dag.topological_op_nodes():
            if not isinstance(nd.op, Delay):
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
                continue

            dag_qubits = nd.qargs

            # initial qubit delays are left unchanged, i.e., left as delays without
            # inserting a decoupling sequence
            if self._skip_reset_qubits:
                pred = next(dag.predecessors(nd))
                if isinstance(pred, DAGInNode) or isinstance(pred.op, Reset):
                    for q in dag_qubits:
                        new_dag.apply_operation_back(Delay(nd.op.duration), [q], [])
                    continue

            start_time = self.property_set["node_start_time"][nd]
            end_time = start_time + nd.op.duration

            # 1) color each qubit in this delay instruction
            coloring = self._get_wire_coloring(dag, nd, qubit_index_map, adjacency_map)

            # 2) insert the actual DD sequences
            physical_qubits = [qubit_index_map[q] for q in dag_qubits]
            for physical_qubit, dag_qubit in zip(physical_qubits, dag_qubits):

                # check the X gate on the active qubit is compatible with pulse alignment
                if physical_qubit not in checked_gate_lengths:
                    x_duration = self._target.durations().get("x", physical_qubit)
                    if x_duration % self._alignment != 0:
                        raise TranspilerError(
                            f"X gate length on qubit {dag_qubit} is {x_duration} which is not "
                            f"an integer multiple of the pulse alignment {self._alignment}."
                        )
                    checked_gate_lengths.add(physical_qubit)

                color = coloring[physical_qubit]
                dd_sequence, spacing = _get_orthogonal_sequence(order=color)

                # check if DD can be applied or if there is not enough time
                dd_sequence_duration = sum(
                    self._durations.get(gate, physical_qubit) for gate in dd_sequence
                )
                slack = nd.op.duration - dd_sequence_duration
                slack_fraction = slack / nd.op.duration
                if 1 - slack_fraction >= self._skip_threshold:  # dd doesn't fit
                    new_dag.apply_operation_back(Delay(nd.op.duration), [dag_qubit], [])
                    continue

                # compute actual spacings in between the delays, taking into account
                # the pulse alignment restriction of the hardware
                taus = self._constrain_spacing(spacing, slack)

                # validate DD sequence and compute the global phase it adds (if any)
                sequence_gphase = _validate_dd_sequence(dd_sequence)

                # apply the DD gates
                # tau has one more entry than the gate sequence
                for tau, gate in itertools.zip_longest(taus, dd_sequence):
                    if tau > 0:
                        new_dag.apply_operation_back(Delay(tau), [dag_qubit])
                    if gate is not None:
                        new_dag.apply_operation_back(gate, [dag_qubit])
                new_dag.global_phase = _mod_2pi(new_dag.global_phase + sequence_gphase)

        return new_dag

    def _get_wire_coloring(self, dag, delay_node, qubit_index_map, adjacency_map):
        start_time = self.property_set["node_start_time"][delay_node]
        end_time = start_time + delay_node.op.duration
        dag_qubits = delay_node.qargs
        physical_qubits = [qubit_index_map[q] for q in dag_qubits]
        neighborhood = set(
            [neighbor for q in physical_qubits for neighbor in adjacency_map.neighbors(q)]
        )
        neighborhood |= set(physical_qubits)
        coloring = {q: None for q in neighborhood}

        # first color qubits that are ctrl/tgt of CX/ECR, in this neighborhood & this time interval
        for q in neighborhood:
            dag_q = dag.qubits[q]
            for q_node in dag.nodes_on_wire(dag_q, only_ops=True):
                # check if the operation occurs during the delay
                adj_start_time = self.property_set["node_start_time"][q_node]
                adj_end_time = adj_start_time + q_node.op.duration
                if (
                    adj_start_time < end_time
                    and adj_end_time > start_time
                    and isinstance(q_node.op, (CXGate, ECRGate))
                ):
                    # set coloring to 0 if ctrl, and to 1 if tgt
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

        return coloring

    def _constrain_spacing(self, spacing, slack):
        def _constrained_length(values):
            return self._alignment * np.floor(values / self._alignment)

        taus = _constrained_length(slack * np.asarray(spacing))
        unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
        middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
        to_middle = _constrained_length(unused_slack)
        taus[middle_index] += to_middle  # now we add up to original delay duration
        if unused_slack - to_middle:
            taus[-1] += unused_slack - to_middle

        return taus


def _get_orthogonal_sequence(order: int) -> (list[Gate], list[float]):
    """Return a DD sequence of given order, where different orders are orthogonal."""
    if order == 0:
        dd_sequence = [XGate(), XGate()]
        spacing = [1 / 2, 1 / 2, 0]
    elif order == 1:
        dd_sequence = [XGate(), XGate()]
        spacing = [1 / 4, 1 / 2, 1 / 4]
    elif order == 2:
        dd_sequence = [XGate(), XGate(), XGate(), XGate()]
        spacing = [1 / 4, 1 / 4, 1 / 4, 1 / 4, 0]
    else:
        raise NotImplementedError(f"Order {order} is not implemented.")

    return dd_sequence, spacing


def _validate_dd_sequence(dd_sequence: list[Gate]) -> float:
    """Check the DD sequence is valid and return the global phase it adds."""
    num_pulses = len(dd_sequence)
    if num_pulses != 1:
        if num_pulses % 2 != 0:
            raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
        noop = np.eye(2)
        for gate in dd_sequence:
            noop = noop.dot(gate.to_matrix())
        if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
            raise TranspilerError("The DD sequence does not make an identity operation.")
        return np.angle(noop[0][0])


def _mod_2pi(angle: float, atol: float = 0):
    """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped
