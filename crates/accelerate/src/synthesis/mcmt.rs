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
use qiskit_circuit::{circuit_data::CircuitData, Operation, Qubit, StandardGate};
use smallvec::SmallVec;

fn cx_v_chain(
    controls: Vec<usize>,
    auxiliaries: Vec<usize>,
    reversed: bool,
) -> impl Iterator<Item = (StandardGate, SmallVec, SmallVec)> {
    let n = len(controls) - 1; // number of chain elements
    let indices = std::iter::once((controls[0], controls[1], auxiliaries[0]))
        .chain((0..n - 1).map(|i| (controls[i + 2], auxiliaries[i], auxiliaries[i + 1])));
    let v_chain = indices.map(|(ctrl1, ctrl2, target)| {
        (
            StandardGate::CCX_GATE,
            smallvec![],
            smallvec![
                Qubit(*ctrl1 as u32),
                Qubit(*ctrl2 as u32),
                Qubit(*target as u32)
            ],
        )
    });
    if reversed {
        v_chain.rev()
    }
    v_chain
}

/// Implement multi-control, multi-target of a single-qubit gate using a V-chain with
/// (num_ctrl_qubits - 1) auxiliary qubits.
/// ``controlled_gate`` here must already be the controlled operation, e.g. if we
/// call MCMT of X, then it must be a CX gate. This is because I currently don't know how to
/// nicely map the single-qubit gate to it's controlled version.
pub fn mcmt_v_chain(
    controlled_gate: Operation,
    num_ctrl_qubits: usize,
    num_target_qubits: usize,
) -> CircuitData {
    let controls = (0..num_ctrl_qubits).collect_vec::<Vec<usize>>();
    let num_qubits = 2 * num_ctrl_qubits - 1 + num_target_qubits;
    let auxiliaries = (num_ctrl_qubits + num_target_qubits..num_qubits).collect_vec::<Vec<usize>>();
    let down_chain = cx_v_chain(controls, auxiliaries, False);
    let up_chain = cx_v_chain(controls, auxiliaries, True);
    let targets = (0..num_target_qubits).map(|i| (controlled_gate, ))
}
