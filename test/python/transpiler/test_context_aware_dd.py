from ddt import ddt, data
import math
import itertools
import numpy as np
from qiskit.circuit import QuantumCircuit, Delay
from qiskit.circuit.library import SXGate, CXGate, XGate, CZGate, ECRGate
from qiskit.transpiler import PassManager, Target, InstructionProperties
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    CombineAdjacentDelays,
    DynamicalDecouplingMulti,
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    ApplyLayout,
    PadDelay,
)

from qiskit_ibm_runtime.fake_provider import FakeHanoiV2


from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestContextAwareDD(QiskitTestCase):
    """Test context-aware dynamical decoupling."""

    def setUp(self):
        super().setUp()
        num_qubits = 5

        # gate times in terms of dt
        self.dt = 1e-9
        self.t_cx = 1e3
        self.t_sx = 10
        self.t_x = 20

        # set up an idealistic target to test context-aware DD in a clean setting
        # (if I don't also add realistic settings I should be scolded)
        target = Target(num_qubits=num_qubits, dt=1e-9)
        # bidirectional linear next neighbor
        linear_topo = [(i, i + 1) for i in range(num_qubits - 1)]
        linear_topo += [tuple(reversed(connection)) for connection in linear_topo]
        # CX, SX and X gate durations (somewhat sensible durations and errors chosen)
        cx_props = {
            connection: InstructionProperties(duration=self.t_cx * self.dt, error=1e-2)
            for connection in linear_topo
        }
        sx_props = {
            (i,): InstructionProperties(duration=self.t_sx * self.dt, error=1e-4)
            for i in range(num_qubits)
        }
        x_props = {
            (i,): InstructionProperties(duration=self.t_x * self.dt, error=1e-4)
            for i in range(num_qubits)
        }
        target.add_instruction(CXGate(), cx_props)
        target.add_instruction(ECRGate(), cx_props)  # re-use CX props for ECR
        target.add_instruction(CZGate(), cx_props)  # re-use CX props for CZ
        target.add_instruction(SXGate(), sx_props)
        target.add_instruction(XGate(), x_props)
        target.add_instruction(Delay(1), sx_props)  # support delays, duration does not matter here

        self.toy_target = target

    def simple_setting(self):
        simple = QuantumCircuit(5)
        simple.sx(simple.qubits)
        simple.barrier()
        simple.cx(1, 2)
        simple.barrier()
        simple.cx(0, 1)
        simple.cx(3, 4)

        return simple

    def paper_setting(self):
        # circuit from the paper
        circuit = QuantumCircuit(6)
        circuit.barrier()
        circuit.cx(1, 0)
        circuit.cx(3, 4)
        circuit.barrier()
        circuit.cx(1, 2)
        circuit.cx(4, 5)
        circuit.barrier()
        circuit.cx(1, 2)

        return circuit

    def test_full(self):
        """Test the full workflow on a simple circuit and a concrete reference."""
        target = self.toy_target
        durations = target.durations()
        dd = PassManager(
            [
                CombineAdjacentDelays(target),
                DynamicalDecouplingMulti(target),
            ]
        )

        circuit = self.simple_setting()
        schedule_pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        pm = schedule_pm + dd

        ref = QuantumCircuit(5)
        ref.sx(ref.qubits)
        ref.barrier()
        apply_delay_sequence(ref, 0, self.t_cx, durations, order=1)  # ctrl sequence
        ref.cx(1, 2)
        apply_delay_sequence(ref, 3, self.t_cx, durations, order=0)  # tgt sequence
        apply_delay_sequence(ref, 4, self.t_cx, durations, order=1)  # lowest seq. orthogonal to tgt

        ref.barrier()
        ref.cx(0, 1)
        apply_delay_sequence(ref, 2, self.t_cx, durations, order=2)  # ctrl+tgt sequence
        ref.cx(3, 4)

        circuit = pm.run(circuit)
        self.assertEqual(circuit, ref)

    def test_simple_combination(self):
        """Test combination on an explicit circuit."""
        target = self.toy_target
        layout = list(range(5))
        dd = PassManager(
            [
                CombineAdjacentDelays(target),
            ]
        )
        circuit = self.simple_setting()
        schedule_pm = _get_schedule_pm(target, layout)
        pm = schedule_pm + dd

    @data(True, False)
    def test_skip_initial_delays(self, skip_initial):
        """Test initial delays are skipped.

           delays here since not after a reset
                    v
        q_0: ──■───────
             ┌─┴─┐
        q_1: ┤ X ├──■──
             └───┘┌─┴─┐
        q_2: ─────┤ X ├
               ^  └───┘
             no delay since after qubit initialization

        """
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        target = self.toy_target
        dd = PassManager(
            [
                CombineAdjacentDelays(target),
                DynamicalDecouplingMulti(target, skip_reset_qubits=skip_initial),
            ]
        )
        schedule_pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        pm = schedule_pm + dd
        circuit = pm.run(circuit)

        # a single ctrl-specific decoupling sequence with 2 X gates if initial delays are skipped,
        # otherwise 4 sequences à 2 X gates due to the target being on 5 qubits
        expected_x = 2 if skip_initial else 8
        self.assertEqual(circuit.count_ops().get("x", 0), expected_x)

    def test_2q_gate_combos(self):
        """Test the ctrl/tgt specific behavior for CX/ECR and default for others (like CZ).

        There are specific sequence for control/target/control+target spectator qubits for
        CX and ECR. Other gates do not get special sequences.

                     ┌────┐ ░      ┌──────┐
            q_0 -> 0 ┤ √X ├─░──────┤0     ├──────────
                     ├────┤ ░      │   ?  │
            q_1 -> 1 ┤ √X ├─░──────┤1     ├──────────
                     ├────┤ ░      └──────┘
            q_2 -> 2 ┤ √X ├─░── test this sequence ──
                     ├────┤ ░      ┌──────┐
            q_3 -> 3 ┤ √X ├─░──────┤0     ├──────────
                     ├────┤ ░      │   ?  │
            q_4 -> 4 ┤ √X ├─░──────┤1     ├──────────
                     └────┘ ░      └──────┘

        """
        target = self.toy_target
        dd = PassManager(
            [
                CombineAdjacentDelays(target),
                DynamicalDecouplingMulti(target),
            ]
        )
        schedule_pm = _get_schedule_pm(target, list(range(5)))
        pm = schedule_pm + dd

        gates = ["cx", "ecr", "cz"]
        for top in gates:
            for bottom in gates:
                if top in ["cx", "ecr"]:
                    if bottom in ["cx", "ecr"]:
                        # squeezed between control and target
                        order = 2
                    else:
                        # other gate (CZ) has no specific behavior, use target-specific sequence
                        order = 0
                else:  # top gate has no specific behavior
                    if bottom in ["cx", "ecr"]:
                        # use ctrl-specific sequence
                        order = 1
                    else:  # both gates do not have special behavior, use 0
                        order = 0

                # construct circuit with specified gates
                circuit = QuantumCircuit(5)
                circuit.sx(circuit.qubits)
                circuit.barrier()
                getattr(circuit, top)(0, 1)
                getattr(circuit, bottom)(3, 4)
                circuit = pm.run(circuit)

                # compute reference
                ref = QuantumCircuit(5)
                ref.sx(ref.qubits)
                ref.barrier()
                getattr(ref, top)(0, 1)
                apply_delay_sequence(ref, 2, self.t_cx, target.durations(), order)
                getattr(ref, bottom)(3, 4)

                with self.subTest(top=top, bottom=bottom):
                    self.assertEqual(circuit, ref)

    def test_threshold_skipping(self):
        """Test skipping of delays that are below the threshold."""
        skip_threshold = 0.6  # arbitrary threshold below 1

        target = self.toy_target
        dd = PassManager(
            [
                DynamicalDecouplingMulti(target, skip_threshold=skip_threshold),
            ]
        )
        schedule_pm = _get_schedule_pm(target, [0])
        pm = schedule_pm + dd

        epsilon = 0.01  # ask a mathematician for a formal definition of epsilon
        dd_duration = 2 * self.t_x
        for exceed_threshold in [True, False]:
            # go above threshold once and below once
            if exceed_threshold:
                delay = math.floor(dd_duration / (skip_threshold + epsilon))
            else:
                delay = math.ceil(dd_duration / (skip_threshold - epsilon))

            circuit = QuantumCircuit(1)
            circuit.sx(0)
            circuit.delay(delay, 0)
            circuit.sx(0)

            circuit = pm.run(circuit)

            with self.subTest(exceed_threshold=exceed_threshold):
                expected_x = 0 if exceed_threshold else 2
                self.assertEqual(circuit.count_ops().get("x", 0), expected_x)

    @data(4, 20)  # X gate length is integer multiple of these values
    def test_pulse_alignment(self, alignment):
        """Test setting the pulse alignment.

        Pulses should only start at integer multiples of the allowed pulse alignment.
        """
        target = self.toy_target
        dd = PassManager(
            [
                DynamicalDecouplingMulti(
                    target, pulse_alignment=alignment, skip_reset_qubits=False
                ),
            ]
        )
        schedule_pm = _get_schedule_pm(target, [0])

        pm = schedule_pm + dd
        circuit = QuantumCircuit(1)
        circuit.delay(100, 0)
        circuit = pm.run(circuit)

        x_times = [
            time for gate, time in pm.property_set["node_start_time"].items() if gate.op.name == "x"
        ]

        self.assertTrue((np.asarray(x_times) % alignment == 0).all())

    def test_invalid_pulse_alignment(self):
        """Test an error is raised if the X gate length is not compatible with the pulse alignment."""
        target = self.toy_target
        dd = PassManager(
            [
                DynamicalDecouplingMulti(
                    target, pulse_alignment=self.t_x + 1, skip_reset_qubits=False
                ),
            ]
        )
        schedule_pm = _get_schedule_pm(target, [0])

        pm = schedule_pm + dd
        circuit = QuantumCircuit(1)
        circuit.delay(100, 0)

        with self.assertRaises(TranspilerError):
            _ = pm.run(circuit)

    def test_orthogonal_sequences(self):
        """Check orthogonality of sequences up to the supported order."""

        # get the DD sequences and check for the smallest time frame
        dd_sequences = [
            np.asarray(DynamicalDecouplingMulti.get_orthogonal_sequence(order))
            for order in range(DynamicalDecouplingMulti.MAX_ORDER + 1)
        ]
        # we must exclude 0, which is sometimes used as padding in the end
        smallest_unit = np.min([np.min(row[np.where(row > 0)]) for row in dd_sequences])

        # expand the DD times to vectors with an entry +1 if the qubit is in + state
        # and a -1 if the qubit is in - state
        sign_vectors = [expand_dd_sequence(sequence, smallest_unit) for sequence in dd_sequences]

        # check pairwise orthogonality
        for i, row_i in enumerate(sign_vectors):
            for j, row_j in enumerate(sign_vectors[i + 1 :]):
                with self.subTest(i=i, j=j):
                    self.assertAlmostEqual(np.dot(row_i, row_j), 0)

    def test_exceed_highest_order(self):
        """Check an error is raised if we query a sequence with order too high."""

        order_threshold = 7  # this is the order not supported anymore
        kranka = Target(num_qubits=order_threshold, dt=self.dt)  # a target with insane connectivity
        coupling_web = itertools.combinations(list(range(order_threshold)), 2)
        cx_props = {
            connection: InstructionProperties(duration=self.t_cx * self.dt, error=1e-2)
            for connection in coupling_web
        }
        x_props = {
            (i,): InstructionProperties(duration=self.t_x * self.dt, error=1e-4)
            for i in range(kranka.num_qubits)
        }
        kranka.add_instruction(CXGate(), cx_props)
        kranka.add_instruction(XGate(), x_props)
        kranka.add_instruction(Delay(1), x_props)

        circuit = QuantumCircuit(order_threshold)
        circuit.x(circuit.qubits)
        circuit.barrier()
        circuit.delay(1000)  # note this should be long enough to fit the DD sequences

        dd = PassManager(
            [
                CombineAdjacentDelays(kranka),
                DynamicalDecouplingMulti(kranka, skip_reset_qubits=False),
            ]
        )
        schedule_pm = _get_schedule_pm(kranka, list(range(kranka.num_qubits)))
        pm = schedule_pm + dd

        with self.assertRaises(TranspilerError):
            _ = pm.run(circuit)


def _get_schedule_pm(target, initial_layout):
    durations = target.durations()

    schedule_pm = PassManager(
        [
            SetLayout(initial_layout),
            FullAncillaAllocation(target),
            EnlargeWithAncilla(),
            ApplyLayout(),
            ALAPScheduleAnalysis(durations, target),
            PadDelay(target=target),
        ]
    )

    return schedule_pm


def apply_delay_sequence(circuit, qubit, timespan, durations, order):
    if order == 0:
        dt = (timespan - 2 * durations.get("x", qubit)) / 2
        for _ in range(2):
            circuit.delay(dt, qubit)
            circuit.x(qubit)
    elif order == 1:
        reduced_timespan = timespan - 2 * durations.get("x", qubit)
        circuit.delay(reduced_timespan / 4, qubit)
        circuit.x(qubit)
        circuit.delay(reduced_timespan / 2, qubit)
        circuit.x(qubit)
        circuit.delay(reduced_timespan / 4, qubit)
    elif order == 2:
        dt = (timespan - 4 * durations.get("x", qubit)) / 4
        for _ in range(4):
            circuit.delay(dt, qubit)
            circuit.x(qubit)

    return circuit


def expand_dd_sequence(sequence, unit):
    """Expand durations in terms of the unit.

    E.g. in the unit is 1/8 and sequence is [1/4, 1/2, 1/4] expand to [++----++].
    """
    signs = []
    sign = 1

    for timespan in sequence:
        num_items = int(timespan / unit)
        signs += num_items * [sign]
        sign *= -1

    return signs
