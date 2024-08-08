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

use crate::QiskitError;
use pyo3::PyErr;
use qiskit_circuit::slice::PySequenceIndex;
use std::iter;

fn _combinations(n: u32, repetitions: u32) -> Vec<Vec<u32>> {
    if repetitions == 1 {
        (0..n).map(|index| vec![index]).collect()
    } else {
        let mut result = Vec::new();
        for indices in _combinations(n, repetitions - 1) {
            let last_element = indices[indices.len() - 1];
            for index in last_element + 1..n {
                let mut extended_indices = indices.clone();
                extended_indices.push(index);
                result.push(extended_indices);
            }
        }
        result
    }
}

pub fn full(num_qubits: u32, block_size: u32) -> impl Iterator<Item = Vec<u32>> {
    // this should be equivalent to itertools.combinations(list(range(n)), m)
    _combinations(num_qubits, block_size).into_iter()
}

/// Return the qubit indices for linear entanglement.
/// For a block_size of ``m``, this is defined as [(0..m-1), (1..m), (2..m+1), ..., (n-m..n-1)]
pub fn linear(num_qubits: u32, block_size: u32) -> impl DoubleEndedIterator<Item = Vec<u32>> {
    (0..num_qubits - block_size + 1)
        .map(move |start_index| (start_index..start_index + block_size).collect())
}

/// Return the qubit indices for a reversed linear entanglement.
pub fn reverse_linear(num_qubits: u32, block_size: u32) -> impl Iterator<Item = Vec<u32>> {
    linear(num_qubits, block_size).rev()
}

/// Return the qubit indices for linear entanglement.
/// For a block_size of ``m`` on ``n`` qubits, this is defined as
/// [(0..m-1), (1..m), (2..m+1), ..., (n-m..n-1), (n-m+1, ..., n-1, 0)]
pub fn circular(num_qubits: u32, block_size: u32) -> Box<dyn Iterator<Item = Vec<u32>>> {
    if block_size == 1 || num_qubits == block_size {
        Box::new(linear(num_qubits, block_size))
    } else {
        // linear(num_qubits, block_size)
        let closing_link = (num_qubits - block_size + 1..num_qubits)
            .chain(iter::once(0))
            .collect();
        Box::new(iter::once(closing_link).chain(linear(num_qubits, block_size)))
    }
}

pub fn pairwise(num_qubits: u32) -> impl Iterator<Item = Vec<u32>> {
    // for Python-folks (like me): pairwise is equal to linear[::2] + linear[1::2]
    linear(num_qubits, 2)
        .step_by(2)
        .chain(linear(num_qubits, 2).skip(1).step_by(2))
}

pub fn shift_circular_alternating(
    num_qubits: u32,
    block_size: u32,
    offset: usize,
) -> Box<dyn Iterator<Item = Vec<u32>>> {
    // index at which we split the circular iterator
    let split =
        PySequenceIndex::convert_idx(-(offset as isize), (num_qubits - block_size + 2) as usize)
            .unwrap();
    let shifted = circular(num_qubits, block_size)
        .skip(split)
        .chain(circular(num_qubits, block_size).take(split));
    if offset % 2 == 0 {
        Box::new(shifted)
    } else {
        // if the offset is odd, reverse the indices inside the qubit block (e.g. turn CX
        // gates upside down)
        Box::new(shifted.map(|indices| indices.into_iter().rev().collect()))
    }
}

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
pub fn get_entanglement(
    num_qubits: u32,
    block_size: u32,
    entanglement: &str,
    offset: usize,
) -> Result<Box<dyn Iterator<Item = Vec<u32>>>, PyErr> {
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
        "full" => Ok(Box::new(full(num_qubits, block_size))),
        "linear" => Ok(Box::new(linear(num_qubits, block_size))),
        "reverse_linear" => Ok(Box::new(reverse_linear(num_qubits, block_size))),
        "sca" => Ok(shift_circular_alternating(num_qubits, block_size, offset)),
        "circular" => Ok(circular(num_qubits, block_size)),
        "pairwise" => Ok(Box::new(pairwise(num_qubits))),
        _ => Err(QiskitError::new_err(format!(
            "Unsupported entanglement: {}",
            entanglement
        ))),
    }
}
