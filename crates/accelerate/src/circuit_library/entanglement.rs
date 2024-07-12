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
use std::iter;

fn _combinations(n: usize, repetitions: usize) -> Vec<Vec<usize>> {
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

pub fn full(num_qubits: usize, block_size: usize) -> impl Iterator<Item = Vec<usize>> {
    // this should be equivalent to itertools.combinations(list(range(n)), m)
    _combinations(num_qubits, block_size).into_iter()
}

/// Return the qubit indices for linear entanglement.
/// For a block_size of ``m``, this is defined as [(0..m-1), (1..m), (2..m+1), ..., (n-m..n-1)]
pub fn linear(num_qubits: usize, block_size: usize) -> impl DoubleEndedIterator<Item = Vec<usize>> {
    (0..num_qubits - block_size + 1)
        .map(move |start_index| (start_index..start_index + block_size).collect())
}

/// Return the qubit indices for a reversed linear entanglement.
pub fn reverse_linear(num_qubits: usize, block_size: usize) -> impl Iterator<Item = Vec<usize>> {
    linear(num_qubits, block_size).rev()
}

/// Return the qubit indices for linear entanglement.
/// For a block_size of ``m`` on ``n`` qubits, this is defined as
/// [(0..m-1), (1..m), (2..m+1), ..., (n-m..n-1), (n-m+1, ..., n-1, 0)]
pub fn circular(num_qubits: usize, block_size: usize) -> Box<dyn Iterator<Item = Vec<usize>>> {
    if block_size == 1 || num_qubits == block_size {
        Box::new(linear(num_qubits, block_size))
    } else {
        // linear(num_qubits, block_size)
        let closing_link = (num_qubits - block_size + 1..num_qubits)
            .chain(iter::once(0))
            .collect();
        Box::new(linear(num_qubits, block_size).chain(iter::once(closing_link)))
    }
}

pub fn pairwise(num_qubits: usize) -> impl Iterator<Item = Vec<usize>> {
    // for Python-folks (like me): pairwise is equal to linear[::2] + linear[1::2]
    linear(num_qubits, 2)
        .step_by(2)
        .chain(linear(num_qubits, 2).skip(1).step_by(2))
}

pub fn shift_circular_alternating(
    num_qubits: usize,
    block_size: usize,
    offset: usize,
) -> Box<dyn Iterator<Item = Vec<usize>>> {
    // index at which we split the circular iterator
    let split = num_qubits - block_size + 1 - offset;
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
