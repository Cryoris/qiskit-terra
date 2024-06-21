from qiskit.circuit import QuantumCircuit, Delay
from qiskit.circuit.library import SXGate, CXGate, XGate
from qiskit.transpiler import PassManager, Target, InstructionProperties, InstructionDurations
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


class TestContextAwareDD(QiskitTestCase):
    """Test context-aware dynamical decoupling."""

    def setUp(self):
        super().setUp()
        num_qubits = 6
        target = Target(num_qubits=num_qubits, dt=1e-9)
        # bidirectional linear next neighbor
        linear_topo = [(i, i + 1) for i in range(num_qubits - 1)] + [(3, 5)]
        linear_topo += [tuple(reversed(connection)) for connection in linear_topo]
        cx_props = {
            connection: InstructionProperties(duration=1e-6, error=1e-2)
            for connection in linear_topo
        }
        sx_props = {
            (i,): InstructionProperties(duration=1e-8, error=1e-4) for i in range(num_qubits)
        }
        x_props = {
            (i,): InstructionProperties(duration=2e-8, error=1e-4) for i in range(num_qubits)
        }
        target.add_instruction(CXGate(), cx_props)
        target.add_instruction(SXGate(), sx_props)
        target.add_instruction(XGate(), x_props)
        target.add_instruction(Delay(1), sx_props)

        self.toy_target = target

    def simple_setting(self):
        simple = QuantumCircuit(4)
        simple.sx(simple.qubits)
        simple.cx(0, 1)
        simple.cx(1, 2)
        simple.cx(2, 3)

        initial_layout = [0, 1, 4, 7]

        return simple, initial_layout

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

        initial_layout = [0, 1, 4, 7, 10, 12]

        return circuit, initial_layout

    def test_simple(self):
        """Test full workflow.

        TODO check the final circuit has expected structure
        """
        # backend = FakeHanoiV2()
        # target = backend.target
        target = self.toy_target

        durations = target.durations()
        schedule_analysis = ASAPScheduleAnalysis(durations, target)
        dd = PassManager(
            [
                CombineAdjacentDelays(target),
                schedule_analysis,  # should not be necessary!
                DynamicalDecouplingMulti(target),
                schedule_analysis,  # should not be necessary!
            ]
        )

        circuit, initial_layout = self.paper_setting()
        # circuit, initial_layout = self.simple_setting()
        schedule_pm = _get_schedule_pm(target, list(range(target.num_qubits)))
        # schedule_pm = _get_schedule_pm(target, initial_layout)

        pm = schedule_pm + dd

        circuit = pm.run(circuit)
        print("\n", circuit.draw(idle_wires=False))

    # def test_combine(self):
    #     """Test adjacent delays are correctly combined.

    #     TODO for different instruction durations, check the the correct blocks are found
    #     """
    #     backend = FakeHanoiV2()
    #     target = backend.target

    #     durations = backend.instruction_durations
    #     schedule_analysis = ALAPScheduleAnalysis(durations, target)

    #     combine = PassManager(
    #         [
    #             CombineAdjacentDelays(target),
    #             schedule_analysis,  # should not be necessary!
    #         ]
    #     )
    #     circuit, initial_layout = self.simple_setting()
    #     schedule_pm = _get_schedule_pm(target, initial_layout)
    #     pm = schedule_pm + combine

    #     circuit = pm.run(circuit)

    #     print("\n", circuit.draw(idle_wires=True))

    #     for op in circuit.data:
    #         if op.operation.name == "delay":
    #             print(op.operation, len(op.qubits))


class TestMultiDD(QiskitTestCase):
    """Test coloring and insertion of DD sequences given a circuit with Delays."""

    def setUp(self):
        super().setUp()

    def test_wire_coloring(self):
        pass


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
