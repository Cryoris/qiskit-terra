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

use num_complex::Complex64;
use qiskit_accelerate::sparse_observable::SparseObservable;

/// Construct the zero observable (without any terms).
///
/// Example:
///
///     SparseObservable* zero = obs_zero(100);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// Construct the identity observable.
///
/// Example:
///
///     SparseObservable* identity = obs_identity(100);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// Multiply the observable by a complex coefficient.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseObservable* result = obs_multiply(obs, 2);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_multiply(
    observable: &SparseObservable,
    coeff: Complex64,
) -> *mut SparseObservable {
    let result = observable * coeff;
    Box::into_raw(Box::new(result))
}

/// Add two observables.
///
/// Example:
///     
///     SparseObservable* left = obs_identity(100);
///     SparseObservable* right = obs_zero(100);
///     SparseObservable* result = obs_add(left, right);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_add(
    left: &SparseObservable,
    right: &SparseObservable,
) -> *mut SparseObservable {
    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// Calculate the canonical representation of the observable.
///
/// Example:
///
///     SparseObservable* iden = obs_identity(100);
///     SparseObservable* two = obs_add(iden, iden);
///
///     let double tol = 1e-6;
///     SparseObservable* canonical = obs_canonicalize(two);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_canonicalize(
    observable: &SparseObservable,
    tol: f64, // no optional arguments in C -- welcome to the ancient past
) -> *mut SparseObservable {
    let result = observable.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// Copy the observable.
///
/// Example:
///
///     SparseObservable* original = obs_identity(100);
///     SparseObservable* copied = obs_copy(original);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_copy(observable: &SparseObservable) -> *mut SparseObservable {
    let copied = observable.clone();
    Box::into_raw(Box::new(copied))
}

/// Deallocate the observable.
///
/// Memory deallocation is user responsibility. Every constructed observable
/// must be deallocated manually to avoid memory leakage.
///
/// Example:
///
///     SparseObservable* obs = obs_zero(100);
///     obs_deallocate(obs);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_deallocate(observable: &mut SparseObservable) {
    unsafe {
        let _ = Box::from_raw(observable);
    }
}

/// Get the number of terms in the observable.
///
/// Example:
///
///     SparseObservable* obs = obs_identity(100);
///     int num_terms = obs_num_terms(obs);  // 1
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_num_terms(observable: &SparseObservable) -> u64 {
    observable.num_terms() as u64
}

/// Get the number of qubits the observable is defined on.
///
/// Example:
///
///     SparseObservable* obs = obs_identity(100);
///     int num_qubits = obs_num_qubits(obs);  // 100
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_num_qubits(observable: &SparseObservable) -> u32 {
    observable.num_qubits() as u32
}

/// Print the observable.
///
/// Example:
///
///     SparseObservable* obs = obs_identity(100);
///     obs_print(obs);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_print(observable: &SparseObservable) {
    println!("{:?}", observable);
}
