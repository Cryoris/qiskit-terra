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

/// @ingroup PauliTermVec
/// A Pauli term vector, containing ``(index, bit_term)`` tuples.
#[derive(Debug, Clone)]
pub struct PauliTermVec {
    // We store the data as two separate vectors, which simplified constructing
    // SparseTerms later on, which expect this format.
    indices: Vec<u32>,
    bit_terms: Vec<BitTerm>,
}

/// @ingroup PauliTerm
/// A struct representing a (Pauli, qubit index) tuple.
#[repr(C)]
pub struct PauliTerm {
    bit_term: BitTerm,
    index: u32,
}

/// @ingroup PauliTerm
/// Free the Pauli term.
///
/// @param A pointer to the Pauli term struct.
///
/// Example:
///
///     SparseObservable *obs = obs_zero(100);
///     PauliTermVec *paulis = paulis_new();
///     paulis_push(paulis, BitTerm_X, 99);  // push X_99 onto the vector
///     obs_push_consume(obs, paulis, 2.0);  // add the Pauli term 2 * X_99 to the observable
///
///     SparseTerm *term = obs_term(obs, 0);  // get the 0th Pauli term in the observable
///     PauliTerm *pauli = obsterm_pauli(term, 0);  // get the 0th Pauli in the Pauli term
///     printf("Bit term: %i, Index: %i", pauli->bit_term, pauli->index);
///
///     pauli_free(pauli);  // deallocate the struct
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn pauli_free(pauli: &mut PauliTerm) {
    unsafe {
        let _ = Box::from_raw(pauli);
    }
}

/// @ingroup PauliTermVec
/// Create a new Pauli term vector.
///
/// @return A pointer to an empty Pauli term vector.
///
/// Example:
///
///     PauliTermVec *paulis = paulis_new();
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn paulis_new() -> *mut PauliTermVec {
    let paulis = PauliTermVec {
        indices: Vec::new(),
        bit_terms: Vec::new(),
    };
    Box::into_raw(Box::new(paulis))
}

/// @ingroup PauliTermVec
/// Create a new Pauli term vector, with a given capacity.
///
/// @param capacity The capacity to allocate for the vector.
///
/// @return A pointer to an empty Pauli term vector.
///
/// Example:
///
///     uint64_t capacity = 10;
///     PauliTermVec *paulis = paulis_with_capacity(capacity);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn paulis_with_capacity(capacity: u64) -> *mut PauliTermVec {
    let paulis = PauliTermVec {
        indices: Vec::with_capacity(capacity as usize),
        bit_terms: Vec::with_capacity(capacity as usize),
    };
    Box::into_raw(Box::new(paulis))
}

/// @ingroup PauliTermVec
/// Free the Pauli term vector.
///
/// @param paulis A pointer to the index vector to be freed.
///
/// Example:
///
///     PauliTermVec *paulis = paulis_new();
///     paulis_push(paulis, BitTerm_Z, 2);
///     paulis_free(paulis);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn paulis_free(paulis: &mut PauliTermVec) {
    unsafe {
        let _ = Box::from_raw(paulis);
    }
}

/// @ingroup PauliTermVec
/// Push a new ``(bit_term, index)`` tuple onto the Pauli term vector.
///
/// @param paulis A pointer to the Pauli term vector.
/// @param bit_term The bit term to add.
/// @param index The index the bit term acts on.
///
/// Example:
///
///     PauliTermVec *paulis = paulis_new();
///     paulis_push(paulis, BitTerm_Z, 2);  // push Z_2 onto the vector
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn paulis_push(paulis: &mut PauliTermVec, bit_term: BitTerm, index: u32) {
    paulis.bit_terms.push(bit_term);
    paulis.indices.push(index);
}

/// @ingroup SparseObservable
/// Construct the zero observable (without any terms).
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// Example:
///
///     SparseObservable *zero = obs_zero(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup SparseObservable
/// Construct the identity observable.
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// Example:
///
///     SparseObservable *identity = obs_identity(100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup SparseObservable
/// Deallocate the observable.
///
/// Memory deallocation is user responsibility. Every constructed observable
/// must be deallocated manually to avoid memory leakage.
///
/// @param obs A pointer to the observable to free.
///
/// Example:
///
///     SparseObservable *obs = obs_zero(100);
///     obs_free(obs);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_free(obs: &mut SparseObservable) {
    unsafe {
        let _ = Box::from_raw(obs);
    }
}

/// @ingroup SparseObservable
/// @brief Add a term to the observable by copy.
///
/// @param obs A pointer to the observable to which the term is added.
/// @param paulis The Pauli term vector to add to the observable.
/// @param coeff The coefficient of the term.
///
/// Example:
///
///     u_int32_t num_qubits = 100;
///     SparseObservable *obs = obs_zero(num_qubits);
///
///     complex double coeff = 1;
///
///     PauliTermVec *paulis = paulis_with_capacity(3);
///     paulis_push(paulis, BitTerm_X, 0);
///     paulis_push(paulis, BitTerm_Y, 1);
///     paulis_push(paulis, BitTerm_Z, 2);
///
///     obs_push_copy(obs, bits, indices, &coeff);  // push the term, without consuming the Pauli term
///
///     paulis_free(paulis);  // manually free the Pauli term vector
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_push_copy(
    obs: &mut SparseObservable,
    paulis: &PauliTermVec,
    coeff: &Complex64,
) {
    let term = SparseTerm::new(
        obs.num_qubits(),
        *coeff, // safe to dereference, because Complex64 implements Copy
        paulis.bit_terms.clone().into_boxed_slice(),
        paulis.indices.clone().into_boxed_slice(),
    )
    .unwrap();

    obs.add_term(term.view()).unwrap();
}

/// @ingroup SparseObservable
/// @brief Add a term to the observable and deallocate the memory for the indices and bit terms.
///
/// @param obs A pointer to the observable to which the term is added.
/// @param paulis The Pauli term vector to add to the observable.
/// @param coeff The coefficient of the term.
///
/// @warning Panics if an index in the Pauli term is greater equal than the number of qubits.
///
/// Example:
///
///     u_int32_t num_qubits = 100;
///     SparseObservable *obs = obs_zero(num_qubits);
///
///     complex double coeff = 1;
///
///     PauliTermVec *paulis = paulis_with_capacity(3);
///     paulis_push(paulis, BitTerm_X, 0);
///     paulis_push(paulis, BitTerm_Y, 1);
///     paulis_push(paulis, BitTerm_Z, 2);
///
///     obs_push_consume(obs, paulis, coeff);  // paulis are deallocated
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_push_consume(
    obs: &mut SparseObservable,
    paulis: &mut PauliTermVec,
    coeff: &Complex64,
) {
    // we take ownership of the memory and let the variables go out of scope
    // after this function, consuming the ``paulis`` variable
    let paulis = unsafe { Box::from_raw(paulis) };

    let term = SparseTerm::new(
        obs.num_qubits(),
        *coeff,
        paulis.bit_terms.into_boxed_slice(),
        paulis.indices.into_boxed_slice(),
    )
    .unwrap(); // TODO handle error

    obs.add_term(term.view()).unwrap(); // TODO handle error
}

/// @ingroup SparseObservable
/// Get an observable term.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
///
/// @return A pointer to a sparse term.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     // out-of-bounds indices will fail
///     // SparseTerm *will_fail = obs_term(obs, 1);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_term(obs: &SparseObservable, index: u64) -> *mut SparseTerm {
    // We could also add a read-only view on the SparseTermView,
    // whose lifetime would implicitly be bound to the observable.
    // For now, we'll only provide a copy, as the Python interface does.
    let term = obs.term(index as usize).to_term();
    Box::into_raw(Box::new(term))
}

/// @ingroup SparseObservable
/// Multiply the observable by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseObservable *result = obs_multiply(obs, 2);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_multiply(obs: &SparseObservable, coeff: &Complex64) -> *mut SparseObservable {
    let result = obs * (*coeff);
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Add two observables.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// @return A pointer to the result ``left + right``.
///
/// Example:
///
///     SparseObservable *left = obs_identity(100);
///     SparseObservable *right = obs_zero(100);
///     SparseObservable *result = obs_add(left, right);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_add(
    left: &SparseObservable,
    right: &SparseObservable,
) -> *mut SparseObservable {
    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Calculate the canonical representation of the observable.
///
/// @param obs A pointer to the observable.
/// @param tol The tolerance below which coefficients are considered to be zero.
///
/// @return The canonical representation of the observable.
///
/// Example:
///
///     SparseObservable *iden = obs_identity(100);
///     SparseObservable *two = obs_add(iden, iden);
///
///     double tol = 1e-6;
///     SparseObservable *canonical = obs_canonicalize(two);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_canonicalize(
    obs: &SparseObservable,
    tol: f64, // no optional arguments in C -- welcome to the ancient past
) -> *mut SparseObservable {
    let result = obs.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// @ingroup SparseObservable
/// Copy the observable.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to a copy of the observable.
///
/// Example:
///
///     SparseObservable *original = obs_identity(100);
///     SparseObservable *copied = obs_copy(original);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_copy(obs: &SparseObservable) -> *mut SparseObservable {
    let copied = obs.clone();
    Box::into_raw(Box::new(copied))
}

/// @ingroup SparseObservable
/// Compare two observables for equality.
///
/// Note that this does not compare mathematical equality, but data equality. This means
/// that two observables might represent the same observable but not compare as equal.
///
/// @param observable A pointer to one observable.
/// @param other A pointer to another observable.
///
/// @return ``true`` if the observables are equal, ``false`` otherwise.
///
/// Example:
///
///     SparseObservable *observable = obs_identity(100);
///     SparseObservable *other = obs_identity(100);
///     bool are_equal = obs_equal(observable, other);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_equal(observable: &SparseObservable, other: &SparseObservable) -> bool {
    observable.eq(other)
}

/// @ingroup SparseObservable
/// Get the number of terms in the observable.
///
/// @param observable A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     uint64_t num_terms = obs_num_terms(obs);  // num_terms==1
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_num_terms(observable: &SparseObservable) -> u64 {
    observable.num_terms() as u64
}

/// @ingroup SparseObservable
/// Get the number of qubits the observable is defined on.
///
/// @param observable A pointer to the observable.
///
/// @return The number of qubits the observable is defined on.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     uint32_t num_qubits = obs_num_qubits(obs);  // 100
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_num_qubits(observable: &SparseObservable) -> u32 {
    observable.num_qubits()
}

/// @ingroup SparseObservable
/// Print the observable.
///
/// @param term A pointer to the ``SparseObservable`` to print.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     obs_print(obs);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_print(observable: &SparseObservable) {
    println!("{:?}", observable);
}

/// @ingroup SparseTerm
/// Deallocate the term.
///
/// The term is **not** automatically deallocated if the observable it
/// is coming from is freed.
///
/// @param term A pointer to the ``SparseTerm`` to deallocate.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///
///     obs_free(obs);  // term is still allocated!
///     obsterm_free(term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_free(term: &mut SparseTerm) {
    unsafe {
        let _ = Box::from_raw(term);
    }
}

/// @ingroup SparseTerm
/// Print a sparse term.
///
/// @param term A pointer to the ``SparseTerm`` to print.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     obsterm_print(term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_print(term: &SparseTerm) {
    println!("{:?}", term);
}

/// @ingroup SparseTerm
/// Get the coefficient of a sparse term.
///
/// @param term A pointer to the ``SparseTerm`` whose coefficient is returned.
///
/// @return The complex coefficient of the sparse term.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     complex double coeff = obsterm_coeff(term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_coeff(term: &SparseTerm) -> Complex64 {
    term.coeff()
}

/// @ingroup SparseTerm
/// Get the number of qubits the sparse term is defined on.
///
/// @param term A pointer to the ``SparseTerm`` whose number of qubits is returned.
///
/// @return The number of qubits the sparse term is defined on.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     uint32_t num_qubits = obsterm_num_qubits(term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_num_qubits(term: &SparseTerm) -> u32 {
    term.num_qubits()
}

/// @ingroup SparseTerm
/// Get the number of non-identity (nni) Paulis in the sparse term.
///
/// @param term A pointer to the ``SparseTerm``.
///
/// @return The number of non-identity Paulis.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     uint32_t nni = obsterm_nni(term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obsterm_nni(term: &SparseTerm) -> u32 {
    // the length can be at most equal to the number of qubits, thus u32 is enough
    term.indices().len() as u32
}

/// @ingroup SparseTerm
/// Get the (Pauli, qubit index) tuple inside term.
///
/// @param term A pointer to the ``SparseTerm``.
/// @param index The index inside the sparse term.
///
/// @return The Pauli and qubit index it acts on as struct.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm *term = obs_term(obs, 0);
///     uint32_t nni = obsterm_nni(term);
///
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
