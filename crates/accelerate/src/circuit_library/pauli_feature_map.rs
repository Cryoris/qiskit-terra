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

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PySequence;
use pyo3::types::PyString;
use pyo3::PyErr;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{add_param, multiply_param, rmultiply_param, Param, StandardGate};
use qiskit_circuit::Qubit;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;

use crate::circuit_library::entanglement;

const PI2: f64 = PI / 2.;

type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn pauli_evolution<'a>(
    py: Python<'a>,
    pauli: &'a str,
    indices: Vec<u32>,
    time: Param,
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    let qubits = indices.iter().map(|i| Qubit(*i)).collect_vec();
    // get pairs of (pauli, qubit) that are active, i.e. that are not the identity
    let active_paulis = pauli
        .chars()
        .zip(qubits)
        .filter(|(p, _)| *p != 'i')
        .collect_vec();

    // if there are no paulis, return an empty iterator -- this case here is also why we use
    // a Box<Iterator>, otherwise the compiler will complain that we return empty one time and
    // a chain another time
    if active_paulis.len() == 0 {
        return Box::new(std::iter::empty::<StandardInstruction>());
    }
    // get the basis change: x -> HGate, y -> RXGate(pi/2), z -> nothing
    let basis_change = active_paulis
        .clone()
        .into_iter()
        .filter(|(p, _)| *p != 'z')
        .map(|(p, q)| match p {
            'x' => (StandardGate::HGate, smallvec![], smallvec![q]),
            'y' => (
                StandardGate::RXGate,
                smallvec![Param::Float(PI2)],
                smallvec![q],
            ),
            _ => unreachable!(),
        });
    // get the inverse basis change
    let inverse_basis_change = basis_change.clone().map(|(gate, _, qubit)| match gate {
        StandardGate::HGate => (gate, smallvec![], qubit),
        StandardGate::RXGate => (gate, smallvec![Param::Float(-PI2)], qubit),
        _ => unreachable!(),
    });
    // get the CX chain down to the target rotation qubit
    let chain_down = active_paulis
        .clone()
        .into_iter()
        .map(|(_, q)| q)
        .tuple_windows()
        .map(|(ctrl, target)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));
    // get the CX chain up (cannot use chain_down.rev since tuple_windows is not double ended)
    let chain_up = active_paulis
        .clone()
        .into_iter()
        .rev()
        .map(|(_, q)| q)
        .tuple_windows()
        .map(|(target, ctrl)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target]));
    // get the RZ gate on the last qubit
    let last_qubit = active_paulis.last().unwrap().1;
    let z_rotation = std::iter::once((
        StandardGate::RZGate,
        smallvec![multiply_param(&time, 2.0, py)],
        smallvec![last_qubit],
    ));
    // and finally chain everything together
    Box::new(
        basis_change
            .chain(chain_down)
            .chain(z_rotation)
            .chain(chain_up)
            .chain(inverse_basis_change),
    )
}

// TODO add: data_map_func: Optional[Callable[[np.ndarray], float]] = None,
// TODO let entanglement be a list
#[pyfunction]
#[pyo3(signature = (feature_dimension, parameters, *, reps=1, entanglement="full", paulis=None, alpha=2.0, insert_barriers=false ))]
pub fn pauli_feature_map(
    py: Python,
    feature_dimension: u32,
    parameters: Bound<PyAny>,
    reps: usize,
    entanglement: &str,
    paulis: Option<&Bound<PySequence>>,
    alpha: f64,
    insert_barriers: bool,
) -> Result<CircuitData, PyErr> {
    let paulis = paulis.map_or_else(
        || Ok(PyList::new_bound(py, vec!["z", "zz"])), // default list is ["z", "zz"]
        PySequenceMethods::to_list,
    )?;
    let pauli_strings = &paulis
        .iter()
        .map(|el| {
            (*el.downcast::<PyString>()
                .expect("Error unpacking the ``paulis`` argument"))
            .to_string()
        })
        .collect_vec();

    let parameter_vector = parameters
        .iter()?
        .map(|el| Param::extract_no_coerce(&el.expect("no idea man")).unwrap())
        .collect_vec();

    let evo = _get_evolution(
        py,
        feature_dimension,
        pauli_strings,
        entanglement,
        &parameter_vector,
        reps,
    );
    CircuitData::from_standard_gates(py, feature_dimension, evo, Param::Float(0.0))
}

fn _pauli_feature_map<'a>(
    py: Python<'a>,
    feature_dimension: u32,
    pauli_strings: &'a Vec<String>,
    entanglement: &'a str,
    parameter_vector: &'a Vec<Param>,
    reps: usize,
) -> impl Iterator<Item = StandardInstruction> + 'a {
    (0..reps).flat_map(move |rep| {
        pauli_strings.into_iter().flat_map(move |pauli| {
            let block_size = pauli.len() as u32;
            let entanglement =
                entanglement::get_entanglement(feature_dimension, block_size, entanglement, rep)
                    .unwrap();
            entanglement.flat_map(move |indices| {
                let angle = _default_reduce(py, parameter_vector, &indices);
                pauli_evolution(py, pauli, indices, angle)
            })
        })
    })
}

fn _default_reduce(py: Python, parameters: &Vec<Param>, indices: &Vec<u32>) -> Param {
    indices.iter().fold(Param::Float(1.0), |acc, i| {
        rmultiply_param(
            acc,
            add_param(&multiply_param(&parameters[*i as usize], -1.0, py), PI, py),
            py,
        )
    })
}
