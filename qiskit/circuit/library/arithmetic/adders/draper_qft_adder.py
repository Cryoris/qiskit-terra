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

"""Compute the sum of two qubit registers using QFT."""

from qiskit.synthesis.arithmetic import adder_d00
from .adder import Adder


class DraperQFTAdder(Adder):
    r"""A circuit that uses QFT to perform in-place addition on two qubit registers.

    For registers with :math:`n` qubits, the QFT adder can perform addition modulo
    :math:`2^n` (with ``kind="fixed"``) or ordinary addition by adding a carry qubits (with
    ``kind="half"``).

    As an example, a non-fixed_point QFT adder circuit that performs addition on two 2-qubit sized
    registers is as follows:

    .. parsed-literal::

         a_0:   ─────────■──────■────────────────────────■────────────────
                         │      │                        │
         a_1:   ─────────┼──────┼────────■──────■────────┼────────────────
                ┌──────┐ │P(π)  │        │      │        │       ┌───────┐
         b_0:   ┤0     ├─■──────┼────────┼──────┼────────┼───────┤0      ├
                │      │        │P(π/2)  │P(π)  │        │       │       │
         b_1:   ┤1 qft ├────────■────────■──────┼────────┼───────┤1 iqft ├
                │      │                        │P(π/2)  │P(π/4) │       │
        cout_0: ┤2     ├────────────────────────■────────■───────┤2      ├
                └──────┘                                         └───────┘

    **References:**

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    [3] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(
        self, num_state_qubits: int, kind: str = "fixed", name: str = "DraperQFTAdder"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            kind: The kind of adder, can be ``'half'`` for a half adder or
                ``'fixed'`` for a fixed-sized adder. A half adder contains a carry-out to represent
                the most-significant bit, but the fixed-sized adder doesn't and hence performs
                addition modulo ``2 ** num_state_qubits``.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        super().__init__(num_state_qubits, name=name)
        circuit = adder_d00(num_state_qubits, kind)
        self.append(circuit.to_gate(), self.qubits)
