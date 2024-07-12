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

use pyo3::prelude::*;

use crate::circuit_library::entanglement;
use crate::QiskitError;
// use qiskit_circuit::circuit_data::CircuitData;

// enum Entanglement {
//     Str,
//     Vec(i64),
// }

// impl Entanglement {
//     pub fn indices(&self, num_qubits: usize) -> Vec<(usize, usize)> {
//         match self {
//             Vec => self.map(|index| index as usize).collect(),
//             str(name) => match name {
//                 "linear" => entanglement::linear(num_qubits),
//                 _ => unreachable!(),
//             },
//         }
//     }
// }

/// Get an entangler map for an arbitrary number of qubits.
///
/// Args:
///     num_qubits: The number of qubits of the circuit.
///     block_size: The number of qubits of the entangling block.
///     entanglement: The entanglement strategy.
///     offset: The block offset, can be used if the entanglements differ per block,
///         for example used in the "sca" mode.
///
/// Returns:
///     The entangler map using mode ``entanglement`` to scatter a block of ``block_size``
///     qubits on ``num_qubits`` qubits.
// if m > n:
//     raise ValueError(
//         "The number of block qubits must be smaller or equal to the number of "
//         "qubits in the circuit."
//     )

// if entanglement == "pairwise" and num_block_qubits > 2:
//     raise ValueError("Pairwise entanglement is not defined for blocks with more than 2 qubits.")

pub fn indices(
    num_qubits: usize,
    block_size: usize,
    entanglement: &str,
    offset: usize,
) -> Result<Box<dyn Iterator<Item = Vec<usize>>>, PyErr> {
    if block_size > num_qubits {
        return Err(QiskitError::new_err(format!(
            "block_size ({}) cannot be larger than number of qubits ({})",
            block_size, num_qubits
        )));
    }

    if entanglement == "pairwise" && block_size != 2 {
        return Err(QiskitError::new_err(format!(
            "block_size ({}) must be 2 for pairwise entanglement",
            block_size
        )));
    }

    match entanglement {
        "full" => Ok(Box::new(entanglement::full(num_qubits, block_size))),
        "linear" => Ok(Box::new(entanglement::linear(num_qubits, block_size))),
        "reverse_linear" => Ok(Box::new(entanglement::reverse_linear(
            num_qubits, block_size,
        ))),
        "sca" => Ok(entanglement::shift_circular_alternating(
            num_qubits, block_size, offset,
        )),
        "circular" => Ok(entanglement::circular(num_qubits, block_size)),
        "pairwise" => Ok(Box::new(entanglement::pairwise(num_qubits))),
        _ => Err(QiskitError::new_err(format!(
            "Unsupported entanglement: {}",
            entanglement
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (num_qubits, entanglement, block_size))]
pub fn n_local(num_qubits: i64, entanglement: &str, block_size: usize) -> () {
    let indices = indices(num_qubits as usize, block_size, entanglement, 0);
    println!("{:?}", indices.unwrap().collect::<Vec<Vec<usize>>>());
}
