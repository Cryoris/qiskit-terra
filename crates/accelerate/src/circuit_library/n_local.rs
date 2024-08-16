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

use core::num;

use pyo3::prelude::*;
use pyo3::types::PyList;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::packed_instruction::PackedOperation;
use rayon::iter::empty;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::{Clbit, Qubit};

use crate::circuit_library::entanglement;

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

// fn _rotation_layer(py: Python, out: &CircuitData, rotation_block: &CircuitData) -> () {
//     let num_qubits = out.num_qubits();
//     let block_size = rotation_block.num_qubits();
//     // let qubits: Vec<Qubit> = (0..num_qubits).map(|i| Qubit(i as u32)).collect();
//     // let qubits = out.qubits.map_indices(bits);

//     for i in 0..num_qubits / block_size {
//         let start_index = (i * block_size) as NodeIndex;
//         for (instruction, qargs, _) in rotation_block.iter() {
//             let indices = qargs.iter().map(|qubit| qubit.0);
//             let new_qubits: Vec<NodeIndex> = indices
//                 .map(|local_index| start_index + local_index)
//                 .collect();
//             // let new_qubits = out.qubits.map_indices([]);
//             let new_clbits: Vec<NodeIndex> = Vec::new();
//             let new_instruction = instruction.clone().op;
//             let circuit_instruction = CircuitInstruction::new(
//                 py,
//                 new_instruction,
//                 new_qubits,
//                 new_clbits,
//                 smallvec![],
//                 None,
//             )
//             .clone()
//             .into_py(py);

//             out.append(
//                 py,
//                 &circuit_instruction
//                     .downcast_bound::<CircuitInstruction>(py)
//                     .unwrap(),
//                 None,
//             );
//         }
//     }
// }

fn rotation_layer<'a>(
    num_qubits: u32,
    packed_rotations: &'a Vec<PackedOperation>,
) -> impl Iterator<Item = Instruction> + 'a {
    // TODO handle parameterization -- probably with parameter vector creation python side
    (0..num_qubits).flat_map(move |i| {
        packed_rotations.into_iter().map(move |packed_rotation| {
            (
                packed_rotation.clone(),
                smallvec![],
                vec![Qubit(i)],
                vec![] as Vec<Clbit>,
            )
        })
    })
}

#[pyfunction]
#[pyo3(signature = (num_qubits, rotation_blocks, entanglement, reps=3))]
pub fn n_local(
    py: Python,
    num_qubits: i64,
    rotation_blocks: &Bound<PyAny>,
    // rotation_block: CircuitData,
    // entanglement_block: CircuitData,
    entanglement: &Bound<PyAny>,
    reps: isize,
) -> PyResult<CircuitData> {
    let py_rotations = rotation_blocks
        .downcast::<PyList>()?
        .into_iter()
        .map(|op| op.extract::<OperationFromPython>())
        .collect::<Result<Vec<_>, _>>()?;
    let packed_rotations = py_rotations
        .iter()
        .map(move |py_rotation| py_rotation.operation.clone())
        .collect();

    let block_size = 2;
    let offset = 0;
    let entanglement =
        entanglement::get_entanglement(num_qubits as u32, block_size, entanglement, offset);

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

    let packed_insts =
        (0..reps as usize).flat_map(|_| rotation_layer(num_qubits as u32, &packed_rotations));

    CircuitData::from_packed_operations(py, num_qubits as u32, 0, packed_insts, Param::Float(0.0))
}
