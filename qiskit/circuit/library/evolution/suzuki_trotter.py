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

"""The Suzuki-Trotter product formula."""

from typing import List, Callable, Optional, Union
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli

from .product_formula import ProductFormula


class SuzukiTrotter(ProductFormula):
    """The (higher order) Suzuki-Trotter product formula.

    References:

        [1]: D. Berry, G. Ahokas, R. Cleve and B. Sanders,
        "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
        `arXiv:quant-ph/0508139 <https://arxiv.org/abs/quant-ph/0508139>`_
    """

    def __init__(
        self,
        reps: int = 1,
        order: int = 2,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: Optional[
            Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]
        ] = None,
    ) -> None:
        """
        Args:
            reps: The number of time steps.
            order: The order of the product formula.
            insert_barriers: If True, insert barriers in between each evolved Pauli.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                "chain", where next neighbor connections are used, or "fountain", where all
                qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single operators.
                Per default, `PauliEvolutionGate` will be used.
        """
        super().__init__(order, reps, insert_barriers, cx_structure, atomic_evolution)

    def synthesize(
        self,
        operators: Union[SparsePauliOp, List[SparsePauliOp]],
        time: Union[float, ParameterExpression],
    ) -> QuantumCircuit:
        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]

        ops_to_evolve = self._recurse(self.order, time / self.reps, pauli_list)

        single_rep = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        for op, coeff in ops_to_evolve:
            # add barriers
            if first_barrier:
                if self.insert_barriers:
                    single_rep.barrier()
            else:
                first_barrier = True

            single_rep.compose(self.atomic_evolution(op, coeff), inplace=True)

        evo = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        for _ in range(self.reps):
            # add barriers
            if first_barrier:
                if self.insert_barriers:
                    single_rep.barrier()
            else:
                first_barrier = True

            evo.compose(single_rep, inplace=True)

        return evo

    @staticmethod
    def _recurse(order, time, pauli_list):
        if order < 1:
            raise ValueError("This bitch empty -- yeet!")

        if order == 1:
            return pauli_list

        elif order == 2:
            halves = [(op, coeff * time / 2) for op, coeff in pauli_list[:-1]]
            full = [(pauli_list[-1][0], time * pauli_list[-1][1])]
            return halves + full + list(reversed(halves))

        else:
            reduction = 1 / (4 - 4 ** (1 / (order - 1)))
            outer = 2 * SuzukiTrotter._recurse(
                order - 2, time=reduction * time, pauli_list=pauli_list
            )
            inner = SuzukiTrotter._recurse(
                order - 2, time=(1 - 4 * reduction) * time, pauli_list=pauli_list
            )
            return outer + inner + outer
