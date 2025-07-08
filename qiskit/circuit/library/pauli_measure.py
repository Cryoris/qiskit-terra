# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Pauli measurement."""

from __future__ import annotations
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.circuit.instruction import Instruction


class PauliMeasure(Instruction):
    """A generic Pauli measurement."""

    def __init__(self, basis: SparsePauliOp | SparseObservable | str) -> None:
        """
        Args:
            basis: The basis to measure in.
        """
        if isinstance(basis, (SparsePauliOp, SparseObservable)):
            as_list = basis.to_sparse_list()
            if len(as_list) != 1:
                raise ValueError(
                    "A Pauli measurement must be instantiated with exactly a single Pauli term."
                )
            self.basis = as_list[0][0]  # the Pauli
            self.indices = as_list[0][1]  # the indices
            num_qubits = basis.num_qubits  # number of qubits the instruction is defined on
        else:
            # if given by str, no sparse representation yet
            self.basis = basis
            self.indices = list(range(0, len(basis)))
            num_qubits = len(basis)

        label = f"Meas({self.basis})"
        super().__init__(
            "PauliMeasure", num_qubits=num_qubits, num_clbits=1, params=[], label=label
        )

    def _define(self):
        # fundamental block, cannot be decomposed further
        return None
