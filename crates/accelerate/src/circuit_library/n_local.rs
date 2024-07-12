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

use std::borrow::Borrow;

use pyo3::prelude::*;
use qiskit_circuit::circuit_instruction::PackedInstruction;
use rustworkx_core::petgraph::adj::NodeIndex;
use rustworkx_core::petgraph::graph::Node;
use smallvec::smallvec;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;

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

fn _rotation_layer(py: Python, out: &CircuitData, rotation_block: &CircuitData) -> () {
    let num_qubits = out.num_qubits();
    let block_size = rotation_block.num_qubits();
    // let qubits: Vec<Qubit> = (0..num_qubits).map(|i| Qubit(i as u32)).collect();
    // let qubits = out.qubits.map_indices(bits);

    for i in 0..num_qubits / block_size {
        let start_index = (i * block_size) as NodeIndex;
        for (instruction, qargs, _) in rotation_block.iter() {
            let indices = qargs.iter().map(|qubit| qubit.0);
            let new_qubits: Vec<NodeIndex> = indices
                .map(|local_index| start_index + local_index)
                .collect();
            // let new_qubits = out.qubits.map_indices([]);
            let new_clbits: Vec<NodeIndex> = Vec::new();
            let new_instruction = instruction.clone().op;
            let circuit_instruction = CircuitInstruction::new(
                py,
                new_instruction,
                new_qubits,
                new_clbits,
                smallvec![],
                None,
            )
            .clone()
            .into_py(py);

            out.append(
                py,
                &circuit_instruction
                    .downcast_bound::<CircuitInstruction>(py)
                    .unwrap(),
                None,
            );
        }
    }
}

#[pyfunction]
#[pyo3(signature = (num_qubits, entanglement))]
pub fn n_local(
    py: Python,
    num_qubits: i64,
    // rotation_block: CircuitData,
    // entanglement_block: CircuitData,
    entanglement: &str,
) -> () {
    let rotation_block = CircuitData::from_standard_gates(
        py,
        1,
        [(StandardGate::HGate, smallvec![], smallvec![Qubit(0)])],
        Param::Float(0.0),
    )
    .unwrap();

    // let entanglement_block = CircuitData::from_standard_gates(
    //     py,
    //     2,
    //     [(
    //         StandardGate::CXGate,
    //         smallvec![],
    //         smallvec![Qubit(0), Qubit(1)],
    //     )],
    //     Param::Float(0.0),
    // )
    // .unwrap();

    // let block_size = entanglement_block.num_qubits();
    // let indices = indices(num_qubits as usize, block_size, entanglement, 0);

    // _rotation_layer(num_qubits as usize, rotation_block)
}
