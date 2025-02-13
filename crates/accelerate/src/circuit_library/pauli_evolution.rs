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

use pyo3::types::{PyList, PyNone, PyString, PyTuple};
use pyo3::{intern, prelude::*};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::{ExtraInstructionAttributes, OperationFromPython};
use qiskit_circuit::operations;
use qiskit_circuit::operations::{multiply_param, radd_param, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};

// custom types for a more readable code
type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Return instructions (using only StandardGate operations) to implement a Pauli evolution
/// of a given Pauli string over a given time (as Param).
///
/// Args:
///     pauli: The Pauli string, e.g. "IXYZ".
///     indices: The qubit indices the Pauli acts on, e.g. if given as [0, 1, 2, 3] with the
///         Pauli "IXYZ", then the correspondence is I_0 X_1 Y_2 Z_3.
///     time: The rotation angle. Note that this will directly be used as input of the
///         rotation gate and not be multiplied by a factor of 2 (that should be done before so
///         that this function can remain Rust-only).
///     phase_gate: If ``true``, use the ``PhaseGate`` instead of ``RZGate`` as single-qubit rotation.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     A pointer to an iterator over standard instructions.
pub fn pauli_evolution<'a>(
    pauli: &'a str,
    indices: Vec<u32>,
    time: Param,
    phase_gate: bool,
    do_fountain: bool,
) -> Box<dyn Iterator<Item = Instruction> + 'a> {
    // ensure the Pauli has no identity terms
    let binding = pauli.to_lowercase(); // lowercase for convenience
    let active = binding
        .as_str()
        .chars()
        .zip(indices)
        .filter(|(pauli, _)| *pauli != 'i');
    let (paulis, indices): (Vec<char>, Vec<u32>) = active.unzip();

    match (phase_gate, indices.len()) {
        (_, 0) => Box::new(std::iter::empty()),
        (false, 1) => Box::new(single_qubit_evolution(paulis[0], indices[0], time)),
        (false, 2) => two_qubit_evolution(paulis, indices, time),
        _ => Box::new(multi_qubit_evolution(
            paulis,
            indices,
            time,
            phase_gate,
            do_fountain,
        )),
    }
}

/// Implement a single-qubit Pauli evolution of a Pauli given as char, on a given index and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
fn single_qubit_evolution(
    pauli: char,
    index: u32,
    time: Param,
) -> Box<dyn Iterator<Item = Instruction>> {
    let qubit = vec![Qubit(index)];

    match pauli {
        'x' => Box::new(std::iter::once((
            StandardGate::RXGate.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        'y' => Box::new(std::iter::once((
            StandardGate::RYGate.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        'z' => Box::new(std::iter::once((
            StandardGate::RZGate.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        _ => Box::new(multi_qubit_evolution(
            vec![pauli],
            vec![index],
            time,
            false,
            false,
        )),
    }
}

/// Implement a 2-qubit Pauli evolution of a Pauli string, on a given indices and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
///
/// If possible, Qiskit's native 2-qubit Pauli rotations are used. Otherwise, the general
/// multi-qubit evolution is called.
fn two_qubit_evolution<'a>(
    pauli: Vec<char>,
    indices: Vec<u32>,
    time: Param,
) -> Box<dyn Iterator<Item = Instruction> + 'a> {
    let qubits = vec![Qubit(indices[0]), Qubit(indices[1])];
    let param: SmallVec<[Param; 3]> = smallvec![time.clone()];
    let paulistring: String = pauli.iter().collect();

    match paulistring.as_str() {
        "xx" => Box::new(std::iter::once((
            StandardGate::RXXGate.into(),
            param,
            qubits,
            vec![],
        ))),
        "zx" => Box::new(std::iter::once((
            StandardGate::RZXGate.into(),
            param,
            qubits,
            vec![],
        ))),
        "yy" => Box::new(std::iter::once((
            StandardGate::RYYGate.into(),
            param,
            qubits,
            vec![],
        ))),
        "zz" => Box::new(std::iter::once((
            StandardGate::RZZGate.into(),
            param,
            qubits,
            vec![],
        ))),
        // Note: the CX modes (do_fountain=true/false) give the same circuit for a 2-qubit
        // Pauli, so we just set it to false here
        _ => Box::new(multi_qubit_evolution(pauli, indices, time, false, false)),
    }
}

/// Implement a multi-qubit Pauli evolution. See ``pauli_evolution`` detailed docs.
fn multi_qubit_evolution(
    pauli: Vec<char>,
    indices: Vec<u32>,
    time: Param,
    phase_gate_for_paulis: bool,
    do_fountain: bool,
) -> impl Iterator<Item = Instruction> {
    let mut control_qubits: Vec<Qubit> = Vec::new(); // indices of projectors
    let mut control_states: Vec<bool> = Vec::new(); // +1 projector (true) or -1 projector (false)
    let mut pauli_qubits: Vec<Qubit> = Vec::new(); // indices of Paulis
    let mut basis_change: Vec<Instruction> = Vec::new(); // basis changes for all

    let paulis = ['x', 'y', 'z'];
    let positive = ['+', 'r', '0'];
    let empty_clbits: Vec<Clbit> = Vec::new();

    for (bit_term, index) in pauli.iter().zip(indices.iter()) {
        let q = Qubit(*index);
        println!("qubit {:?} bit_term {:?}", q, bit_term);
        match bit_term {
            'x' | '+' | '-' => basis_change.push((
                StandardGate::HGate.into(),
                smallvec![],
                vec![q.clone()],
                empty_clbits.clone(),
            )),
            'y' | 'r' | 'l' => basis_change.push((
                StandardGate::SXGate.into(),
                smallvec![],
                vec![q.clone()],
                empty_clbits.clone(),
            )),
            _ => {}
        };

        if paulis.contains(bit_term) {
            pauli_qubits.push(q);
        } else {
            control_qubits.push(q);
            control_states.push(positive.contains(bit_term));
        }
    }

    // get the inverse basis change
    let inverse_basis_change: Vec<Instruction> = basis_change
        .iter()
        .map(|(gate, _, qubit, _)| match gate.standard_gate() {
            StandardGate::HGate => (
                StandardGate::HGate.into(),
                smallvec![],
                qubit.clone(),
                empty_clbits.clone(),
            ),
            StandardGate::SXGate => (
                StandardGate::SXdgGate.into(),
                smallvec![],
                qubit.clone(),
                empty_clbits.clone(),
            ),
            _ => unreachable!("Invalid basis-changing Clifford."),
        })
        .collect();

    // get the CX propagation up to the first qubit, and down
    let (chain_up, chain_down) = match do_fountain {
        true => (
            cx_fountain(pauli_qubits.clone()),
            cx_fountain(pauli_qubits.clone()).rev(),
        ),
        false => (
            cx_chain(pauli_qubits.clone()),
            cx_chain(pauli_qubits.clone()).rev(),
        ),
    };

    // get the Z/phase rotation targeting the first qubit
    let rotation = if pauli_qubits.len() > 0 {
        let params: SmallVec<[Param; 3]> = smallvec![time];
        let base_gate = if phase_gate_for_paulis {
            StandardGate::PhaseGate
        } else {
            StandardGate::RZGate
        };

        let (packed, qubits) = if control_qubits.len() == 0 {
            let gate: PackedOperation = base_gate.into();
            (gate, vec![pauli_qubits[0]])
        } else {
            control_qubits.push(pauli_qubits[0]); // the control_qubits variable is no longer used
            let controlled = add_control(base_gate, &params, &control_states).unwrap();
            (controlled, control_qubits)
        };
        vec![(packed, params, qubits, empty_clbits.clone())]
    } else {
        let params: SmallVec<[Param; 3]> =
            Python::with_gil(|py| smallvec![multiply_param(&time, -0.5, py)]);
        let (packed, qubits) = if control_qubits.len() == 1 {
            let gate: PackedOperation = StandardGate::PhaseGate.into();
            (gate, vec![control_qubits[0]])
        } else {
            println!("control qubits {:?}", control_qubits);
            println!("control states {:?}", control_states);
            let controlled =
                add_control(StandardGate::PhaseGate, &params, &control_states[1..]).unwrap();
            let mut qubits: Vec<Qubit> = Vec::with_capacity(control_qubits.len());
            // qubits.extend_from_slice(&control_qubits[1..]);
            // qubits.push(control_qubits[0]);
            control_qubits.reverse();
            (controlled, control_qubits)
        };
        let inst: Instruction = (packed, params, qubits.clone(), empty_clbits.clone());

        if control_states[0] {
            // sandwich in X gates for the correct projector
            let x: Instruction = (
                StandardGate::XGate.into(),
                smallvec![],
                vec![*qubits.last().unwrap()],
                empty_clbits.clone(),
            );
            vec![x.clone(), inst, x]
        } else {
            vec![inst]
        }
    };

    // let first_qubit = pauli_qubits.first().unwrap();
    // let z_rotation = std::iter::once((
    //     if phase_gate {
    //         StandardGate::PhaseGate
    //     } else {
    //         StandardGate::RZGate
    //     },
    //     smallvec![time],
    //     smallvec![*first_qubit],
    // ));

    // and finally chain everything together
    basis_change
        .into_iter()
        .chain(chain_down)
        .chain(rotation.into_iter())
        .chain(chain_up)
        .chain(inverse_basis_change)
}

/// Implement a Pauli evolution circuit.
///
/// The Pauli evolution is implemented as a basis transformation to the Pauli-Z basis,
/// followed by a CX-chain and then a single Pauli-Z rotation on the last qubit. Then the CX-chain
/// is uncomputed and the inverse basis transformation applied. E.g. for the evolution under the
/// Pauli string XIYZ we have the circuit
///
///        ┌───┐      ┌───┐┌───────┐┌───┐┌───┐
///     0: ┤ H ├──────┤ X ├┤ Rz(2) ├┤ X ├┤ H ├────────
///        └───┘      └─┬─┘└───────┘└─┬─┘└───┘
///     1: ─────────────┼─────────────┼───────────────
///        ┌────┐┌───┐  │             │  ┌───┐┌──────┐
///     2: ┤ √X ├┤ X ├──■─────────────■──┤ X ├┤ √Xdg ├
///        └────┘└─┬─┘                   └─┬─┘└──────┘
///     3: ────────■───────────────────────■──────────
///
/// Args:
///     num_qubits: The number of qubits in the Hamiltonian.
///     sparse_paulis: The Paulis to implement. Given in a sparse-list format with elements
///         ``(pauli_string, qubit_indices, rz_rotation_angle)``. An element of the form
///         ``("XIYZ", [0,1,2,3], 2)``, for example, is interpreted in terms of qubit indices as
///         X_q0 I_q1 Y_q2 Z_q3 and will use a RZ rotation angle of 2.
///     insert_barriers: If ``true``, insert a barrier in between the evolution of individual
///         Pauli terms.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     Circuit data for to implement the evolution.
#[pyfunction]
#[pyo3(name = "pauli_evolution", signature = (num_qubits, sparse_paulis, insert_barriers=false, do_fountain=false))]
pub fn py_pauli_evolution(
    num_qubits: i64,
    sparse_paulis: &Bound<PyList>,
    insert_barriers: bool,
    do_fountain: bool,
) -> PyResult<CircuitData> {
    let py = sparse_paulis.py();
    let num_paulis = sparse_paulis.len();
    let mut paulis: Vec<String> = Vec::with_capacity(num_paulis);
    let mut indices: Vec<Vec<u32>> = Vec::with_capacity(num_paulis);
    let mut times: Vec<Param> = Vec::with_capacity(num_paulis);
    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false; // keep track of whether we modified the phase

    for el in sparse_paulis.iter() {
        let tuple = el.downcast::<PyTuple>()?;
        let pauli = tuple.get_item(0)?.downcast::<PyString>()?.to_string();
        let time = Param::extract_no_coerce(&tuple.get_item(2)?)?;

        if pauli.as_str().chars().all(|p| p == 'i') {
            global_phase = radd_param(global_phase, time, py);
            modified_phase = true;
            continue;
        }

        paulis.push(pauli);
        times.push(time); // note we do not multiply by 2 here, this is already done Python side!
        indices.push(tuple.get_item(1)?.extract::<Vec<u32>>()?)
    }

    let barrier = (
        PackedOperation::from_standard_instruction(operations::StandardInstruction::Barrier(
            num_qubits as u32,
        )),
        smallvec![],
        (0..num_qubits as u32).map(Qubit).collect(),
        vec![],
    );

    let evos = paulis.iter().enumerate().zip(indices).zip(times).flat_map(
        |(((i, pauli), qubits), time)| {
            let as_packed = pauli_evolution(pauli, qubits, time, false, do_fountain).map(Ok);
            // this creates an iterator containing a barrier only if required, otherwise it is empty
            let maybe_barrier = (insert_barriers && i < (num_paulis - 1))
                .then_some(Ok(barrier.clone()))
                .into_iter();
            as_packed.chain(maybe_barrier)
        },
    );

    // When handling all-identity Paulis above, we added the RZ rotation angle as global phase,
    // meaning that we have implemented of exp(i 2t I). However, what we want it to implement
    // exp(-i t I). To only use a single multiplication, we apply a factor of -0.5 here.
    // This is faster, in particular as long as the parameter expressions are in Python.
    if modified_phase {
        global_phase = multiply_param(&global_phase, -0.5, py);
    }

    CircuitData::from_packed_operations(py, num_qubits as u32, 0, evos, global_phase)
}

/// Build a CX chain over the active qubits. E.g. with q_1 inactive, this would return
///
///                    ┌───┐
///     q_0: ──────────┤ X ├
///                    └─┬─┘
///     q_1: ────────────┼──
///               ┌───┐  │
///     q_2: ─────┤ X ├──■──
///          ┌───┐└─┬─┘
///     q_3: ┤ X ├──■───────
///          └─┬─┘
///     q_4: ──■────────────
///
fn cx_chain(qubits: Vec<Qubit>) -> Box<dyn DoubleEndedIterator<Item = Instruction>> {
    let num_terms = qubits.len();
    if num_terms < 2 {
        return Box::new(std::iter::empty());
    }

    Box::new(
        (0..num_terms - 1)
            .map(move |i| (qubits[i], qubits[i + 1]))
            .map(|(target, ctrl)| {
                (
                    StandardGate::CXGate.into(),
                    smallvec![],
                    vec![ctrl, target],
                    vec![],
                )
            }),
    )
}

/// Build a CX fountain over the active qubits. E.g. with q_1 inactive, this would return
///
///         ┌───┐┌───┐┌───┐
///    q_0: ┤ X ├┤ X ├┤ X ├
///         └─┬─┘└─┬─┘└─┬─┘
///    q_1: ──┼────┼────┼──
///           │    │    │
///    q_2: ──■────┼────┼──
///                │    │
///    q_3: ───────■────┼──
///                     │
///    q_4: ────────────■──
///
fn cx_fountain(qubits: Vec<Qubit>) -> Box<dyn DoubleEndedIterator<Item = Instruction>> {
    let num_terms = qubits.len();
    if num_terms < 2 {
        return Box::new(std::iter::empty());
    }

    let first_qubit = qubits[0];
    Box::new((1..num_terms).rev().map(move |i| {
        let ctrl = qubits[i];
        (
            StandardGate::CXGate.into(),
            smallvec![],
            vec![ctrl, first_qubit],
            vec![],
        )
    }))
}

fn add_control(
    gate: StandardGate,
    params: &[Param],
    control_state: &[bool],
) -> PyResult<PackedOperation> {
    Python::with_gil(|py| {
        let extra_attrs = ExtraInstructionAttributes::default();
        let pygate = gate.create_py_op(py, Some(params), &extra_attrs)?;
        let num_controls = control_state.len();
        let py_control_state = PyString::new(
            py,
            control_state
                .iter()
                .map(|is_open| is_open.then(|| '0').unwrap_or_else(|| '1'))
                .collect::<String>()
                .as_str(),
        );
        println!("Control state {:?}", py_control_state);
        let label = PyNone::get(py);
        let controlled_gate = pygate
            .call_method1(
                py,
                intern!(py, "control"),
                (num_controls, label, py_control_state),
            )?
            .extract::<OperationFromPython>(py)?;

        Ok(controlled_gate.operation)
    })
}
