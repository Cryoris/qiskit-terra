# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of the abstract class UCPauliRotGate for uniformly controlled
(also called multiplexed) single-qubit rotations around the Y-axes
(i.e., uniformly controlled R_y rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_y(a_i) is applied to the target qubit.
"""

from .uc_pauli_rot import UCPauliRotGate


class UCRYGate(UCPauliRotGate):
    """
    Uniformly controlled rotations (also called multiplexed rotations).
    The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al.
    (https://arxiv.org/pdf/quant-ph/0406176.pdf)
    """

    def __init__(self, angle_list):
        super().__init__(angle_list, "Y")
