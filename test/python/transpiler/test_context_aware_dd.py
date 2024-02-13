from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes import CombineAdjacentDelays, DynamicalDecouplingMulti
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.compiler import transpile


from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestContextAwareDD(QiskitTestCase):
    """Test context-aware dynamical decoupling."""

    def setUp(self):
        super().setUp()

        simple = QuantumCircuit(3)
        simple.h(simple.qubits)
        simple.cx(0, 1)
        simple.cx(1, 2)
        simple.h(simple.qubits)

        self.simple = simple

        backend = GenericBackendV2(num_qubits=3, calibrate_instructions=True)
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        self.pm = pm

    def test_simple(self):
        scheduled = self.pm.run(self.simple)
        print(scheduled.draw())
        self.assertTrue(False)
