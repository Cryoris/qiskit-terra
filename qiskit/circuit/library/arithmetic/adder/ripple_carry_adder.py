# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two qubit registers using ripple-carry approach."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister


class RippleCarryAdder(QuantumCircuit):
    r"""A ripple-carry circuit to perform in-place addition on two qubit registers.

    Circuit to compute the sum of two qubit registers using the approach from [1].
    Given two equally sized input registers that store quantum states
    :math:`|a\rangle` and :math:`|b\rangle`, performs addition of numbers that
    can be represented by the states, storing the resulting state in-place in the second register:

    .. math::

        |a\rangle |b\rangle \mapsto |a\rangle |a+b\rangle

    Here :math:`|a\rangle` (and correspondingly :math:`|b\rangle`) stands for the direct product
    :math:`|a_n\rangle \otimes |a_{n-1}\rangle \ldots |a_{1}\rangle \otimes |a_{0}\rangle`
    which denotes a quantum register prepared with the value :math:`a = 2^{0}a_{0} + 2^{1}a_{1} +
    \ldots 2^{n}a_{n}` [2].

    As an example, a ripple-carry adder circuit that performs addition on two 3-qubit sized
    registers is as follows:

    .. parsed-literal::

                     ┌──────┐┌──────┐                     ┌──────┐┌──────┐
          input_a_0: ┤0     ├┤2     ├─────────────────────┤2     ├┤0     ├
                     │      ││      │┌──────┐     ┌──────┐│      ││      │
          input_a_1: ┤      ├┤0     ├┤2     ├─────┤2     ├┤0     ├┤      ├
                     │      ││      ││      │     │      ││      ││      │
          input_a_2: ┤      ├┤  MAJ ├┤0     ├──■──┤0     ├┤  UMA ├┤      ├
                     │      ││      ││      │  │  │      ││      ││      │
          input_b_0: ┤1 MAJ ├┤      ├┤  MAJ ├──┼──┤  UMA ├┤      ├┤1 UMA ├
                     │      ││      ││      │  │  │      ││      ││      │
          input_b_1: ┤      ├┤1     ├┤      ├──┼──┤      ├┤1     ├┤      ├
                     │      │└──────┘│      │  │  │      │└──────┘│      │
          input_b_2: ┤      ├────────┤1     ├──┼──┤1     ├────────┤      ├
                     │      │        └──────┘  │  └──────┘        │      │
         carry_in_0: ┤2     ├──────────────────┼──────────────────┤2     ├
                     └──────┘                ┌─┴─┐                └──────┘
        carry_out_0: ────────────────────────┤ X ├────────────────────────
                                             └───┘

    Here *MAJ* and *UMA* gates correspond to the gates introduced in [1]. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is different ordering as compared to Figure 4 in [1], which leads to a different
    drawing of the circuit.

    **References:**

    [1] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
    `arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>`_

    [2] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_
    """

    def __init__(self,
                 num_state_qubits: int,
                 name: str = 'RippleCarryAdder'
                 ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            name: The name of the circuit object.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        qr_a = QuantumRegister(num_state_qubits, name='input_a')
        qr_b = QuantumRegister(num_state_qubits, name='input_b')
        qr_z = QuantumRegister(1, name='carry_out')
        qr_c = AncillaRegister(1, name='carry_in')

        # initialize quantum circuit with register list
        super().__init__(qr_a, qr_b, qr_z, qr_c, name=name)

        # build carry circuit for majority of 3 bits in-place
        # corresponds to MAJ gate in [1]
        qc_maj = QuantumCircuit(3, name='MAJ')
        qc_maj.cnot(0, 1)
        qc_maj.cnot(0, 2)
        qc_maj.toffoli(2, 1, 0)
        qc_instruction_mac = qc_maj.to_instruction()

        # build circuit for reversing carry operation
        # corresponds to UMA gate in [1]
        qc_uma = QuantumCircuit(3, name='UMA')
        qc_uma.toffoli(2, 1, 0)
        qc_uma.cnot(0, 2)
        qc_uma.cnot(2, 1)
        qc_instruction_uma = qc_uma.to_instruction()

        # build ripple-carry adder circuit
        self.append(qc_instruction_mac, [qr_a[0], qr_b[0], qr_c[0]])
        for i in range(num_state_qubits-1):
            self.append(qc_instruction_mac, [qr_a[i+1], qr_b[i+1], qr_a[i]])
        self.cnot(qr_a[-1], qr_z[0])
        for i in reversed(range(num_state_qubits-1)):
            self.append(qc_instruction_uma, [qr_a[i+1], qr_b[i+1], qr_a[i]])
        self.append(qc_instruction_uma, [qr_a[0], qr_b[0], qr_c[0]])
