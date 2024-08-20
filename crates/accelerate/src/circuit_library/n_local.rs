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
use pyo3::types::PyList;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::Param;
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
    parameters: &'a Vec<Vec<Vec<Param>>>,
) -> impl Iterator<Item = Instruction> + 'a {
    // TODO handle parameterization -- probably with parameter vector creation python side
    packed_rotations
        .iter()
        .zip(parameters)
        .flat_map(move |(packed_op, block_params)| {
            (0..num_qubits).zip(block_params).map(move |(i, params)| {
                (
                    packed_op.clone(),
                    SmallVec::from_vec(params.clone()),
                    vec![Qubit(i)],
                    vec![] as Vec<Clbit>,
                )
            })
        })

    // (0..num_qubits).flat_map(move |i| {
    //     packed_rotations.into_iter().map(move |(packed_op, num_params)| {
    //         (
    //             packed_op.clone(),
    //             smallvec![],
    //             vec![Qubit(i)],
    //             vec![] as Vec<Clbit>,
    //         )
    //     })
    // })
}

#[pyfunction]
#[pyo3(signature = (num_qubits, rotation_blocks, reps, entanglement, parameters))]
pub fn n_local(
    py: Python,
    num_qubits: i64,
    rotation_blocks: &Bound<PyAny>,
    // rotation_block: CircuitData,
    // entanglement_block: CircuitData,
    reps: isize,
    entanglement: &Bound<PyAny>,
    parameters: Bound<PyAny>, // take reference?
) -> PyResult<CircuitData> {
    // extract the parameters from the input variable ``parameters``
    // the rotation parameters are given as list[list[list[list[ParameterExpression]]]]
    let parameter_vector: Vec<Vec<Vec<Vec<Param>>>> = parameters
        .downcast::<PyList>()
        .expect("Parameters must be list[list[list[ParameterExpression]]]")
        .into_iter()
        .map(|per_rep| {
            per_rep
                .downcast::<PyList>()
                .expect("Parameters must be list[list[list[ParameterExpression]]]")
                .into_iter()
                .map(|per_block| {
                    per_block
                        .downcast::<PyList>()
                        .expect("Parameters must be list[list[list[ParameterExpression]]]")
                        .into_iter()
                        .map(|per_qubit| {
                            per_qubit
                                .downcast::<PyList>()
                                .expect("")
                                .into_iter()
                                .map(|el| {
                                    Param::extract_no_coerce(&el)
                                        .expect("Error extracting the ParameterExpression.")
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let py_rotations = rotation_blocks
        .downcast::<PyList>()?
        .into_iter()
        .map(|op| op.extract::<OperationFromPython>())
        .collect::<Result<Vec<_>, _>>()?;
    let packed_rotations: Vec<PackedOperation> = py_rotations
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

    let packed_insts = (0..reps as usize).flat_map(|rep| {
        rotation_layer(num_qubits as u32, &packed_rotations, &parameter_vector[rep])
    });

    CircuitData::from_packed_operations(py, num_qubits as u32, 0, packed_insts, Param::Float(0.0))
}
