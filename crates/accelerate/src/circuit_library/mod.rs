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
use pyo3::types::PyTuple;

mod entanglement;
// mod n_local;
mod pauli_feature_map;

#[pyfunction]
#[pyo3(signature = (block_size, num_qubits, entanglement, offset))]
pub fn get_entangler_map<'py>(
    py: Python<'py>,
    block_size: u32,
    num_qubits: u32,
    entanglement: &str,
    offset: usize,
) -> PyResult<Vec<Bound<'py, PyTuple>>> {
    match entanglement::get_entanglement(num_qubits, block_size, entanglement, offset) {
        Ok(entanglement) => Ok(entanglement
            .into_iter()
            .map(|vec| PyTuple::new_bound(py, vec))
            .collect()),
        Err(e) => Err(e),
    }
}

#[pymodule]
pub fn circuit_library(m: &Bound<PyModule>) -> PyResult<()> {
    // m.add_wrapped(wrap_pyfunction!(n_local::n_local))?;
    m.add_wrapped(wrap_pyfunction!(pauli_feature_map::pauli_feature_map))?;
    m.add_wrapped(wrap_pyfunction!(get_entangler_map))?;
    Ok(())
}
