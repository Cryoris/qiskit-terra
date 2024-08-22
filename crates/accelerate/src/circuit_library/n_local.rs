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

use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PyGate, PyInstruction};
use qiskit_circuit::{imports, Clbit, Qubit};

use itertools::izip;

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

fn rotation_layer<'a>(
    py: Python<'a>,
    num_qubits: u32,
    packed_rotations: &'a Vec<(PackedOperation, u32)>,
    parameters: &'a Vec<Vec<Vec<Param>>>,
    skipped_qubits: &'a HashSet<u32>,
) -> impl Iterator<Item = Instruction> + 'a {
    packed_rotations
        .iter()
        .zip(parameters)
        .flat_map(move |((packed_op, size), block_params)| {
            (0..num_qubits)
                .step_by(*size as usize)
                .filter(move |start_idx| {
                    skipped_qubits
                        .is_disjoint(&HashSet::from_iter(*start_idx..(*start_idx + *size)))
                })
                .zip(block_params)
                .map(move |(start_idx, params)| {
                    let rebound = rebind_op(py, packed_op, params).expect("Failed to rebind.");
                    (
                        rebound,
                        SmallVec::from_vec(params.clone()),
                        (0..*size).map(|i| Qubit(start_idx + i)).collect(),
                        vec![] as Vec<Clbit>,
                    )
                })
        })
}

fn entanglement_layer<'a>(
    py: Python<'a>,
    entanglement: &'a Vec<Vec<Vec<u32>>>,
    packend_entanglings: &'a Vec<(PackedOperation, u32)>,
    parameters: &'a Vec<Vec<Vec<Param>>>,
) -> impl Iterator<Item = Instruction> + 'a {
    let zipped = izip!(packend_entanglings, parameters, entanglement);
    zipped.flat_map(move |(packed_op, block_params, block_entanglement)| {
        block_entanglement
            .iter()
            .zip(block_params)
            .map(move |(indices, params)| {
                let rebound = rebind_op(py, &packed_op.0, params).expect("Failed to rebind.");
                (
                    rebound,
                    SmallVec::from_vec(params.clone()),
                    indices.iter().map(|i| Qubit(*i)).collect(),
                    vec![] as Vec<Clbit>,
                )
            })
    })
}

#[pyfunction]
#[pyo3(signature = (num_qubits, reps, rotation_blocks, rotation_parameters, entanglement, entanglement_blocks, entanglement_parameters, insert_barriers, skip_final_rotation_layer, skip_unentangled_qubits))]
pub fn n_local(
    py: Python,
    num_qubits: i64,
    reps: i64,
    rotation_blocks: &Bound<PyAny>,
    rotation_parameters: Bound<PyAny>, // take reference?
    entanglement: &Bound<PyAny>,
    entanglement_blocks: &Bound<PyAny>,
    entanglement_parameters: Bound<PyAny>,
    insert_barriers: bool,
    skip_final_rotation_layer: bool,
    skip_unentangled_qubits: bool,
) -> PyResult<CircuitData> {
    // extract the parameters from the input variable ``parameters``
    // the rotation parameters are given as list[list[list[list[ParameterExpression]]]]
    let rotation_parameters = extract_parameters(rotation_parameters);
    let entanglement_parameters = extract_parameters(entanglement_parameters);

    let packed_rotations = extract_packed_ops(rotation_blocks)?;
    let packed_entanglings = extract_packed_ops(entanglement_blocks)?;

    let entanglement = extract_entanglement(entanglement);

    // Compute the qubits that are skipped in the rotation layer. If this is set,
    // we skip qubits that do not appear in any of the entanglement layers.
    let skipped_qubits = if skip_unentangled_qubits {
        let active: HashSet<&u32> =
            HashSet::from_iter(entanglement.iter().flatten().flatten().flatten());
        HashSet::from_iter((0..num_qubits as u32).filter(|i| !active.contains(i)))
    } else {
        HashSet::new()
    };

    // This is a barrier, if we insert barriers, otherwise an empty vector.
    // This allows to nicely iterate over this variable instead of nasty conditional chains later on.
    let barrier: Vec<Instruction> = match insert_barriers {
        true => vec![get_barrier(py, num_qubits as u32)],
        false => vec![],
    };

    let mut packed_insts: Box<dyn Iterator<Item = Instruction>> =
        Box::new((0..reps as usize).flat_map(|rep| {
            rotation_layer(
                py,
                num_qubits as u32,
                &packed_rotations,
                &rotation_parameters[rep],
                &skipped_qubits,
            )
            .chain(barrier.clone().into_iter())
            .chain(entanglement_layer(
                py,
                &entanglement[rep],
                &packed_entanglings,
                &entanglement_parameters[rep],
            ))
            .chain(barrier.clone().into_iter())
        }));
    if !skip_final_rotation_layer {
        packed_insts = Box::new(packed_insts.chain(rotation_layer(
            py,
            num_qubits as u32,
            &packed_rotations,
            &rotation_parameters[reps as usize],
            &skipped_qubits,
        )))
    }

    CircuitData::from_packed_operations(py, num_qubits as u32, 0, packed_insts, Param::Float(0.0))
}

fn extract_entanglement(py_list: &Bound<PyAny>) -> Vec<Vec<Vec<Vec<u32>>>> {
    py_list
        .downcast::<PyList>()
        .expect("Entanglement must be list[list[list[tuple[int]]]]")
        .into_iter()
        .map(|per_rep| {
            per_rep
                .downcast::<PyList>()
                .expect("Entanglement must be list[list[list[tuple[int]]]]")
                .into_iter()
                .map(|per_block| {
                    per_block
                        .downcast::<PyList>()
                        .expect("Entanglement must be list[list[list[tuple[int]]]]")
                        .into_iter()
                        .map(|connections| {
                            connections
                                .downcast::<PyTuple>()
                                .expect("Entanglement must be list[list[list[tuple[int]]]]")
                                .into_iter()
                                .map(|index| index.extract().expect("Failed getting index."))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn extract_parameters(py_parameters: Bound<PyAny>) -> Vec<Vec<Vec<Vec<Param>>>> {
    py_parameters
        .downcast::<PyList>()
        .expect("Parameters must be list[list[list[list[ParameterExpression]]]]")
        .into_iter()
        .map(|per_rep| {
            per_rep
                .downcast::<PyList>()
                .expect("Parameters must be list[list[list[list[ParameterExpression]]]]")
                .into_iter()
                .map(|per_block| {
                    per_block
                        .downcast::<PyList>()
                        .expect("Parameters must be list[list[list[list[ParameterExpression]]]]")
                        .into_iter()
                        .map(|per_qubit| {
                            per_qubit
                                .downcast::<PyList>()
                                .expect("Parameters must be list[list[list[list[ParameterExpression]]]]")
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
        .collect()
}

fn extract_packed_ops(gatelist: &Bound<PyAny>) -> PyResult<Vec<(PackedOperation, u32)>> {
    let py_ops = gatelist
        .downcast::<PyList>()?
        .into_iter()
        .map(|op| op.extract::<OperationFromPython>())
        .collect::<Result<Vec<_>, _>>()?;
    let packed_rotations: Vec<(PackedOperation, u32)> = py_ops
        .iter()
        .map(move |py_op| {
            let packed_op = &py_op.operation;
            (packed_op.clone(), packed_op.view().num_qubits())
        })
        .collect();
    Ok(packed_rotations)
}

fn get_barrier(py: Python, num_qubits: u32) -> Instruction {
    let barrier_cls = imports::BARRIER.get_bound(py);
    let barrier = barrier_cls
        .call1((num_qubits,))
        .expect("Could not create Barrier Python-side");
    let barrier_inst = PyInstruction {
        qubits: num_qubits,
        clbits: 0,
        params: 0,
        op_name: "barrier".to_string(),
        control_flow: false,
        instruction: barrier.into(),
    };
    (
        barrier_inst.into(),
        smallvec![],
        (0..num_qubits).map(|i| Qubit(i)).collect(),
        vec![] as Vec<Clbit>,
    )
}

fn rebind_op(
    py: Python,
    packed_op: &PackedOperation,
    params: &Vec<Param>,
) -> PyResult<PackedOperation> {
    let out = packed_op.clone();
    match out.view() {
        OperationRef::Standard(_) => {
            Ok(out) // if StandardGate, no rebinding necessary
        }
        OperationRef::Operation(_) => {
            Ok(out) // if Operation, no rebinding necessary
        }
        // for gate and instruction we copy the operation and bind the parameters on
        // the underlying Python object using Instruction._assign_parameters
        OperationRef::Gate(py_gate) => {
            let py_params =
                PyList::new_bound(py, params.iter().map(|p| p.clone().into_py(py))).into_any();
            let copy = py_gate
                .gate
                .call_method1(py, "_assign_parameters", (py_params, false))?;
            let bound_gate = Box::new(PyGate {
                qubits: py_gate.qubits,
                clbits: py_gate.clbits,
                params: py_gate.params,
                op_name: py_gate.op_name.clone(),
                gate: copy,
            });
            Ok(PackedOperation::from_gate(bound_gate))
        }
        OperationRef::Instruction(py_inst) => {
            let py_params =
                PyList::new_bound(py, params.iter().map(|p| p.clone().into_py(py))).into_any();
            let copy =
                py_inst
                    .instruction
                    .call_method1(py, "_assign_parameters", (py_params, false))?;
            let bound_inst = Box::new(PyInstruction {
                qubits: py_inst.qubits,
                clbits: py_inst.clbits,
                params: py_inst.params,
                op_name: py_inst.op_name.clone(),
                control_flow: py_inst.control_flow,
                instruction: copy,
            });
            Ok(PackedOperation::from_instruction(bound_inst))
        }
    }
}
