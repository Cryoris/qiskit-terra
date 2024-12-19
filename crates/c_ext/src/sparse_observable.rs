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
use qiskit_accelerate::sparse_observable::{BitTerm, SparseObservable, SparseTerm};

type IndexVec = Vec<u32>;
type BitTermVec = Vec<BitTerm>;

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn indices_new() -> *mut IndexVec {
    let indices = IndexVec::new();
    Box::into_raw(Box::new(indices))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn indices_with_capacity(capacity: u64) -> *mut IndexVec {
    let indices = IndexVec::with_capacity(capacity as usize);
    Box::into_raw(Box::new(indices))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn indices_push(indices: &mut IndexVec, value: u32) {
    indices.push(value)
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_terms_new() -> *mut BitTermVec {
    let bit_terms = BitTermVec::new();
    Box::into_raw(Box::new(bit_terms))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_terms_with_capacity(capacity: u64) -> *mut BitTermVec {
    let bit_terms = BitTermVec::with_capacity(capacity as usize);
    Box::into_raw(Box::new(bit_terms))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_terms_push(bit_terms: &mut BitTermVec, value: BitTerm) {
    bit_terms.push(value)
}

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

/// Add a term to the observable.
///
/// Example:  TODO
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_push_copy(
    obs: &mut SparseObservable,
    bit_terms: &BitTermVec,
    indices: &IndexVec,
    coeff: Complex64,
) {
    let term = SparseTerm::new(
        obs.num_qubits(),
        coeff,
        bit_terms.clone().into_boxed_slice(),
        indices.clone().into_boxed_slice(),
    )
    .unwrap();

    obs.add_term(term.view()).unwrap();
}

/// Add a term to the observable.
///
/// Example:  TODO
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_push_consume(
    obs: &mut SparseObservable,
    bit_terms: &mut BitTermVec,
    indices: &mut IndexVec,
    coeff: Complex64,
) {
    // we take ownership of the memory and let the variables go out of scope
    // after this function, consuming the variables ``bit_terms`` and ``indices``
    let bit_terms = unsafe { Box::from_raw(bit_terms) };
    let indices = unsafe { Box::from_raw(indices) };

    let term = SparseTerm::new(
        obs.num_qubits(),
        coeff,
        bit_terms.into_boxed_slice(),
        indices.into_boxed_slice(),
    )
    .unwrap(); // TODO handle error

    obs.add_term(term.view()).unwrap(); // TODO handle error
}

/// Get an observable term.
///
/// Example:
///
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_term(obs: &SparseObservable, index: u64) -> *mut SparseTerm {
    // We could also add a read-only view on the SparseTermView,
    // whose lifetime would implicitly be bound to the observable.
    // For now, we'll only provide a copy, as the Python interface does.
    let term = obs.term(index as usize).to_term();
    Box::into_raw(Box::new(term))
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
/// @param term A pointer to the ``SparseObservable`` to print.
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

/// Deallocate the term.
///
/// The term is **not** automatically deallocated if the observable it
/// is coming from is freed.
///
/// @param term A pointer to the ``SparseTerm`` to deallocate.
///
/// Example:
///
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///
///     obs_deallocate(obs);  // term is still allocated!
///     obsterm_deallocate(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_deallocate(term: &mut SparseTerm) {
    unsafe {
        let _ = Box::from_raw(term);
    }
}

/// Print a sparse term.
///
/// @param term A pointer to the ``SparseTerm`` to print.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///     obsterm_print(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_print(term: &SparseTerm) {
    println!("{:?}", term);
}

/// Get the coefficient of a sparse term.
///
/// @param term A pointer to the ``SparseTerm`` whose coefficient is returned.
///
/// @return The complex coefficient of the sparse term.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///     complex double coeff = obsterm_coeff(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_coeff(term: &SparseTerm) -> Complex64 {
    term.coeff()
}

/// Get the number of qubits the sparse term is defined on.
///
/// @param term A pointer to the ``SparseTerm`` whose number of qubits is returned.
///
/// @return The number of qubits the sparse term is defined on.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///     uint32_t num_qubits = obsterm_num_qubits(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_num_qubits(term: &SparseTerm) -> u32 {
    term.num_qubits()
}

/// Get the number of non-identity (nni) Paulis in the sparse term.
///
/// @param term A pointer to the ``SparseTerm``.
///
/// @return The number of non-identity Paulis.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///     uint32_t nni = obsterm_nni(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_nni(term: &SparseTerm) -> u32 {
    // the length can be at most equal to the number of qubits, thus u32 is enough
    term.indices().len() as u32
}

/// A struct representing a (Pauli, qubit index) tuple.
#[repr(C)]
pub struct PauliTerm {
    bit_term: BitTerm,
    index: u32,
}

/// Get the (Pauli, qubit index) tuple inside term.
///
/// @param term A pointer to the ``SparseTerm``.
/// @param index The index inside the sparse term.
///
/// @return The Pauli and qubit index it acts on as struct.
///
/// Example:
///     
///     SparseObservable* obs = obs_identity(100);
///     SparseTerm* term = obs_term(obs, 0);
///     uint32_t nni = obsterm_nni(term);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_pauli(term: &SparseTerm, index: u32) -> *mut PauliTerm {
    let index = index as usize;
    if index >= term.indices().len() {
        panic!("Index out of range.");
    }

    let pauli_term = PauliTerm {
        bit_term: term.bit_terms()[index],
        index: term.indices()[index],
    };
    Box::into_raw(Box::new(pauli_term))
}
