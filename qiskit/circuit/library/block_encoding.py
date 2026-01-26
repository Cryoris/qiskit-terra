import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, PauliGate
from qiskit.quantum_info import SparseObservable


class BlockEncoding(Gate):
    """A block encoding."""

    def __init__(self, operator: SparseObservable):
        super().__init__("block_encoding", operator.num_qubits, params=[operator])

    def validate_parameter(self, parameter):
        if isinstance(parameter, SparseObservable):
            return parameter
        raise TypeError("Unsupported parameter.")

    def _define(self):
        if self._definition is None:
            # we implement Prep^dagger Select Prep for the Hamiltonian
            op = self.params[0]

            # get the padded state preparation implementing |0> -> \sum_i c_i |i>
            prep = state_prep(op.coeffs)

            # build the select oracle implementing |i>|psi> -> |i> P_i|psi>
            oracle = select(op, num_controls=prep.num_qubits)

            controls = QuantumRegister(prep.num_qubits)
            state = QuantumRegister(op.num_qubits)
            encoding = QuantumCircuit(controls, state)
            encoding.append(prep.inverse(), controls)
            encoding.append(oracle, controls[:] + state[:])
            encoding.append(prep, controls)

            self._definition = encoding

        return self._definition


def state_prep(coeffs) -> StatePreparation:
    # pad coeffs to 2^n
    target_len = 2 ** int(np.ceil(np.log2(len(coeffs))))
    padded = np.concatenate(
        (np.asarray(coeffs) / np.linalg.norm(coeffs), np.zeros(target_len - len(coeffs)))
    )
    return StatePreparation(padded, normalize=True)


def pauli_gate_from_term(term) -> PauliGate:
    global_label = ["I"] * term.num_qubits
    for label, i in zip(term.bit_labels(), term.indices):
        global_label[i] = label

    return PauliGate("".join(global_label))


def select(observable, num_controls) -> QuantumCircuit:
    controls = QuantumRegister(num_controls)
    state = QuantumRegister(observable.num_qubits)
    oracle = QuantumCircuit(controls, state)  # controls on top

    for i, term in enumerate(observable):
        pauli_gate = pauli_gate_from_term(term)
        control_state = bin(i)[2:]
        controlled = pauli_gate.control(len(controls), ctrl_state=control_state)
        oracle.append(controlled, controls[:] + state[:])

    return oracle
