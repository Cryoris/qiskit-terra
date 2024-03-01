from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    CombineAdjacentDelays,
    DynamicalDecouplingMulti,
    ALAPScheduleAnalysis,
    ApplyLayout,
    PadDelay,
)
from qiskit.compiler.scheduler import schedule

from qiskit_ibm_runtime.fake_provider import FakeHanoiV2


from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestContextAwareDD(QiskitTestCase):
    """Test context-aware dynamical decoupling."""

    def setUp(self):
        super().setUp()

        simple = QuantumCircuit(4)
        simple.sx(simple.qubits)
        simple.cx(0, 1)
        simple.cx(1, 2)
        simple.cx(2, 3)

        self.simple = simple
        self.backend = FakeHanoiV2()
        initial_layout = [0, 1, 4, 7]
        self.target = self.backend.target

        # would be nice to use but has some weird behavior that 1q gates can be
        # longer than 2q gates
        # backend = GenericBackendV2(num_qubits=3, calibrate_instructions=True)

        durations = self.backend.instruction_durations

        schedule_pm = PassManager(
            [
                SetLayout(initial_layout),
                FullAncillaAllocation(self.target),
                EnlargeWithAncilla(),
                ApplyLayout(),
                ALAPScheduleAnalysis(durations, self.target),
                PadDelay(target=self.target),
            ]
        )
        self.schedule_pm = schedule_pm

    def test_simple(self):
        """Test full workflow.

        TODO check the final circuit has expected structure
        """
        durations = self.backend.instruction_durations
        schedule_analysis = ALAPScheduleAnalysis(durations, self.target)
        dd = PassManager(
            [
                CombineAdjacentDelays(self.target),
                schedule_analysis,  # should not be necessary!
                DynamicalDecouplingMulti(self.target),
                schedule_analysis,  # should not be necessary!
            ]
        )
        pm = self.schedule_pm + dd

        circuit = pm.run(self.simple)
        print(circuit.draw(idle_wires=False))

    def test_combine(self):
        """Test adjacent delays are correctly combined.

        TODO for different instruction durations, check the the correct blocks are found
        """
        durations = self.backend.instruction_durations
        schedule_analysis = ALAPScheduleAnalysis(durations, self.target)
        combine = PassManager(
            [
                CombineAdjacentDelays(self.target),
                schedule_analysis,  # should not be necessary!
            ]
        )
        pm = self.schedule_pm + combine

        circuit = pm.run(self.simple)
        print(circuit.draw(idle_wires=False))
