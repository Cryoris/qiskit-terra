// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::PyResult;
use qiskit_circuit::{
    Qubit,
    circuit_data::CircuitData,
    operations::{Param, StandardGate},
};
use smallvec::smallvec;

// 0 - H - C -----
// 1 ----- X - C -
// 2 --------- X -
pub fn mycircuit() -> PyResult<CircuitData> {
    let num_qubits = 3;
    let instructions = [
        (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
        (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
        (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
    ];
    let global_phase = Param::Float(3.14151415);
    CircuitData::from_standard_gates(num_qubits, instructions, global_phase)
}
