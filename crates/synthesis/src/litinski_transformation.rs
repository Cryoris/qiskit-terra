// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_traits::Zero;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_data::CircuitError;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::PyGate;
use qiskit_circuit::Clbit;
use qiskit_circuit::Qubit;
use std::f64::consts::{FRAC_PI_2, PI};

use qiskit_circuit::imports::{PAULI_EVOLUTION_GATE, PAULI_MEASURE};
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use pyo3::prelude::*;
use rustiq_core::routines::rotation_extraction::extract_rotations;

fn try_rz_to_clifford(angle: f64, tol: f64) -> (&'static str, Option<f64>) {
    let div = angle / FRAC_PI_2;
    let div_rounded = div.round();
    if (div - div_rounded).abs() > tol {
        return ("RZ", Some(angle));
    }
    let multiple = div_rounded as i64;

    if multiple.rem_euclid(2).is_zero() {
        ("Z", None)
    } else {
        if multiple.rem_euclid(4) == 1 {
            ("S", None)
        } else {
            // remainder is 3
            ("Sd", None)
        }
    }
}

#[pyfunction]
pub fn to_pbc(py: Python, circuit: &CircuitData) -> PyResult<CircuitData> {
    // turn the Qiskit circuit into Rustiq's format; that is a vector of (Pauli string, indices)
    // and keep track of the rotation angles in a separate list
    let nq = circuit.num_qubits();
    let mut rustiq_circuit: Vec<(String, Vec<usize>)> = Vec::new();
    let mut angles: Vec<Option<f64>> = Vec::new();
    let mut clbits: Vec<Clbit> = Vec::new();
    let interner = circuit.cargs_interner();

    for inst in circuit.iter() {
        let name = inst.op.name();
        let (rustiq_name, angle) = match name {
            "cx" => ("CX", None),
            "cz" => ("CZ", None),
            "h" => ("H", None),
            "s" => ("S", None),
            "sdg" => ("Sd", None),
            "sx" => ("SqrtX", None),
            "sxdg" => ("SqrtXd", None),
            "x" => ("X", None),
            "y" => ("Y", None),
            "z" => ("Z", None),
            "t" => ("RZ", Some(PI / 8.)),
            "tdg" => ("RZ", Some(-PI / 8.0)),
            "rz" => match inst.params_view()[0] {
                Param::Float(angle) => try_rz_to_clifford(angle, 1e-10),
                _ => unreachable!("RZ must have a parameter"),
            },
            "measure" => ("RZ", None), // handle measure as RZ, marking it with a None angle
            _ => {
                return Err(CircuitError::new_err(format!(
                    "The gate {name:?} is not supported in Rustiq."
                )))
            }
        };

        let qubits: Vec<usize> = circuit
            .get_qargs(inst.qubits)
            .iter()
            .map(|q| q.index())
            .collect();

        if name == "measure" {
            let bits = interner.get(inst.clbits);
            clbits.push(*bits.get(0).expect("No clbit found, expected one!"));
        }

        rustiq_circuit.push((rustiq_name.to_string(), qubits));
        if rustiq_name == "RZ" {
            angles.push(angle);
        }
    }

    // apply the Litinski transformation
    let (rotations, _clifford) = extract_rotations(&rustiq_circuit, nq);

    // rebuild the Qiskit circuit using PauliEvolutionGates and PauliMeasure
    let mut new_circuit = CircuitData::clone_empty_like(&circuit, None)?;

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
    let py_meas_cls = PAULI_MEASURE.get_bound(py);
    let no_clbits: Vec<Clbit> = Vec::new();
    let mut clbit_index = 0usize;

    for ((sign, pauli), angle) in rotations.iter().zip(angles) {
        // sparsify the label
        let (qubits, paulis): (Vec<Qubit>, String) = pauli
            .chars()
            .enumerate()
            .filter(|(_index, p)| *p != 'I')
            .map(|(index, p)| (Qubit(index as u32), p))
            .unzip();

        let py_pauli =
            PySparseObservable::from_label(paulis.chars().rev().collect::<String>().as_str())?;

        match angle {
            Some(angle) => {
                let time = if *sign { -angle } else { angle };
                let py_evo = py_evo_cls.call1((py_pauli, time))?;
                let py_gate = PyGate {
                    qubits: qubits.len() as u32,
                    clbits: 0,
                    params: 1,
                    op_name: "PauliEvolution".to_string(),
                    gate: py_evo.into(),
                };

                new_circuit.push_packed_operation(
                    py_gate.into(),
                    &[Param::Float(time)],
                    &qubits,
                    &no_clbits,
                );
            }
            None => {
                let py_meas = py_meas_cls.call1((py_pauli,))?;
                let py_gate = PyGate {
                    qubits: qubits.len() as u32,
                    clbits: 1,
                    params: 1,
                    op_name: "PauliMeasure".to_string(),
                    gate: py_meas.into(),
                };

                // retrieve the classical bit we measured into (we know it's a single one)
                let clbit = vec![clbits[clbit_index]];
                clbit_index += 1;

                new_circuit.push_packed_operation(py_gate.into(), &[], &qubits, &clbit);
            }
        }
    }

    Ok(new_circuit)
}

pub fn litinski_transformation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(to_pbc))?;
    Ok(())
}
