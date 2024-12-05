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

use qiskit_accelerate::sparse_observable::SparseObservable;

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn hello() {
    println!("Sparse observable will go here!");
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_deallocate(observable: &mut SparseObservable) {
    unsafe {
        let _ = Box::from_raw(observable);
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_print(observable: &mut SparseObservable) {
    println!("{:?}", observable);
}
