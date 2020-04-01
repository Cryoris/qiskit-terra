# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Piecewise-linearly-controlled rotation."""

from typing import List
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.arithmetic import (LinearPauliRotations as LinR,
                                               FixedValueComparator as Comparator)


class PiecewiseLinearPauliRotations(QuantumCircuit):
    """Piecewise-linearly-controlled Pauli rotations.

    For a piecewise linear (not necessarily continuous) function f(x).
    The function f(x) is defined through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints { x_0, ..., x_J } are a subset of [0,  2^n-1], where
    n is the number of state qubits.
    Further on, denote the corresponding slopes and offsets by a_j, b_j respectively.
    Then f(x) is defined as:

        x < x_0            --> f(x) = 0
        x_j <= x < x_{j+1} --> f(x) = a_j * (x - x_j) + b_j

    where we implicitly assume x_{J+1} = 2^n.
    """

    def __init__(self, num_state_qubits: int, breakpoints: List[int], slopes: List[float],
                 offsets: List[float], basis: str = 'Y') -> None:
        """Construct piecewise-linearly-controlled Pauli rotations.

        Args:
            num_state_qubits: The number of qubits representing the state.
            breakpoints: The breakpoints to define the piecewise-linear function.
            slopes: The slopes for different segments of the piecewise-linear function.
            offsets: The offsets for different segments of the piecewise-linear function.
            basis: The type of Pauli rotation ('X', 'Y', 'Z').
        """
        # store parameters
        self.num_state_qubits = num_state_qubits
        self.breakpoints = breakpoints
        self.slopes = slopes
        self.offsets = offsets
        self.basis = basis

        # map slopes and offsets
        self.mapped_slopes = np.zeros(len(breakpoints))
        self.mapped_offsets = np.zeros(len(breakpoints))
        self.mapped_slopes[0] = self.slopes[0]
        self.mapped_offsets[0] = self.offsets[0] - self.slopes[0] * self.breakpoints[0]
        sum_mapped_slopes = 0
        sum_mapped_offsets = 0
        for i in range(1, len(breakpoints)):
            sum_mapped_slopes += self.mapped_slopes[i - 1]
            sum_mapped_offsets += self.mapped_offsets[i - 1]

            self.mapped_slopes[i] = self.slopes[i] - sum_mapped_slopes
            self.mapped_offsets[i] = \
                self.offsets[i] - self.slopes[i] * self.breakpoints[i] - sum_mapped_offsets

        # check whether 0 is contained in breakpoints
        self.contains_zero_breakpoint = np.isclose(0, self.breakpoints[0])

        qr_state = QuantumRegister(num_state_qubits)
        qr_target = QuantumRegister(1)
        qr_ancilla = QuantumRegister(self.num_ancilla_qubits)

        super().__init__(qr_state, qr_target, qr_ancilla)

        self._build(qr_state, qr_target, qr_ancilla)

    def evaluate(self, x: float) -> float:
        """Classically evaluate the piecewise linear rotation.

        Args:
            x: Value to be evaluated at.

        Returns:
            Value of piecewise linear function at x.
        """

        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * self.mapped_slopes[i] +
                                                  self.mapped_offsets[i])

        return y

    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits.

        Returns:
            The number of ancilla qubits in the circuit.
        """
        num_ancilla_qubits = self.num_state_qubits - 1 + len(self.breakpoints)
        if self.contains_zero_breakpoint:
            num_ancilla_qubits -= 1
        return num_ancilla_qubits

    def _build(self, qr_state, qr_target, qr_ancilla):
        # apply comparators and controlled linear rotations
        for i, point in enumerate(self.breakpoints):
            if i == 0 and self.contains_zero_breakpoint:
                # apply rotation
                lin_r = LinR(num_qubits=self.num_state_qubits + 1,
                             slope=self.mapped_slopes[i],
                             offset=self.mapped_offsets[i],
                             basis=self.basis)
                self.append(lin_r.to_gate(), qr_state[:] + qr_target[:])

            else:
                if self.contains_zero_breakpoint:
                    i_compare = i - 1
                else:
                    i_compare = i

                # apply comparator
                comp = Comparator(num_state_qubits=self.num_state_qubits, value=point)
                qr = qr_state[:] + [qr_ancilla[i_compare]]  # add ancilla as compare qubit
                qr_remaining_ancilla = qr_ancilla[i_compare + 1:]  # take remaining ancillas

                self.append(comp.to_gate(),
                            qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])

                # apply controlled rotation
                lin_r = LinR(num_qubits=self.num_state_qubits + 1,
                             slope=self.mapped_slopes[i],
                             offset=self.mapped_offsets[i],
                             basis=self.basis)
                self.append(lin_r.to_gate().control(),
                            [qr_ancilla[i_compare]] + qr_state[:] + qr_target[:])

                # uncompute comparator
                self.append(comp.to_gate().inverse(),
                            qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])
