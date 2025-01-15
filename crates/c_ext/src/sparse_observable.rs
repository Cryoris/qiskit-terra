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

use crate::exit_codes::ExitCode;
use num_complex::Complex64;
use qiskit_accelerate::sparse_observable::{BitTerm, SparseObservable, SparseTermView};
use thiserror::Error;

/// Errors related to C input.
#[derive(Error, Debug)]
pub enum CInputError {
    #[error("Unexpected null pointer.")]
    NullPointerError,
    #[error("Non-aligned memory.")]
    AlignmentError,
}

/// @ingroup SparseTerm
/// A term in a [SparseObservable].
///
/// This term does not own the data, but only contains pointers (and a size) to the
/// bit terms, indices and the coefficient.
#[repr(C)]
pub struct CSparseTerm {
    coeff: *mut Complex64,
    len: usize,
    bit_terms: *mut BitTerm,
    indices: *mut u32,
    num_qubits: u32,
}

impl TryFrom<&CSparseTerm> for SparseTermView<'_> {
    type Error = CInputError;

    fn try_from(value: &CSparseTerm) -> Result<Self, Self::Error> {
        if value.bit_terms.is_null() || value.indices.is_null() {
            return Err(CInputError::NullPointerError);
        }

        // not stable in Rust 1.70 yet
        // if !value.bit_terms.is_aligned() || !value.indices.is_aligned() {
        //     return Err(CInputError::AlignmentError);
        // }

        let bit_terms = unsafe { ::std::slice::from_raw_parts(value.bit_terms, value.len) };
        let indices = unsafe { ::std::slice::from_raw_parts(value.indices, value.len) };

        Ok(SparseTermView {
            num_qubits: value.num_qubits,
            coeff: unsafe { *value.coeff },
            bit_terms,
            indices,
        })
    }
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
/// Add a term to the observable.
///
/// @param obs A pointer to the observable.
/// @param A pointer to the term to add.
///
/// @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
///
/// Example:
///
///     uint32_t num_qubits = 100;
///     SparseObservable *obs = obs_zero(num_qubits);
///
///     complex double coeff = 1;
///     BitTerm bit_terms[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     SparseTerm term = {&coeff, 3, bit_terms, indices, num_qubits};
///
///     obs_add_term(obs, &term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_add_term(obs: &mut SparseObservable, cterm: &CSparseTerm) -> ExitCode {
    let view = match cterm.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    match obs.add_term(view) {
        Ok(_) => ExitCode::Success,
        Err(err) => ExitCode::from(err),
    }
}

/// @ingroup SparseObservable
/// Get an observable term by reference. This can modify the underlying observable.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
/// @param out A pointer to a [CSparseTerm] used to return the observable term.
///
/// @return An exit code.
///
/// Example:
///
///     SparseObservable *obs = obs_identity(100);
///     SparseTerm term;
///     obs_term(obs, 0, &term);
///     // out-of-bounds indices return an error code
///     // int error = obs_term(obs, 12, &term);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_term(
    obs: &mut SparseObservable,
    index: u64,
    out: &mut CSparseTerm,
) -> ExitCode {
    let index = index as usize;
    if index > obs.num_terms() {
        return ExitCode::IndexError;
    }

    out.len = obs.boundaries()[index + 1] - obs.boundaries()[index];
    out.coeff = &mut obs.coeffs_mut()[index];
    out.num_qubits = obs.num_qubits();

    let start = obs.boundaries()[index];
    out.bit_terms = &mut obs.bit_terms_mut()[start];
    out.indices = unsafe { &mut obs.indices_mut()[start] };

    ExitCode::Success
}

/// @ingroup SparseObservable
/// Get a pointer to the coefficients.
///
/// This can be used to read and modify the observable's coefficients.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the coefficients.
///
/// Example:
///
///     SparseObservable *iden = obs_identity(100);
///     complex double *coeffs = obs_coeffs(obs);
///     assert coeffs[0] == 1;
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_coeffs(obs: &mut SparseObservable) -> *mut Complex64 {
    &mut obs.coeffs_mut()[0]
}

/// @ingroup SparseObservable
/// Get a pointer to the indices.
///
/// This can be used to read and modify the observable's indices.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the indices.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_indices(obs: &mut SparseObservable) -> *mut u32 {
    // this is unsafe as we can no longer ensure the indices are within
    // range of the observable's number of qubits
    unsafe { &mut obs.indices_mut()[0] }
}

/// @ingroup SparseObservable
/// Get a pointer to the bit terms.
///
/// This can be used to read and modify the observable's bit terms.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the bit terms.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obs_bit_terms(obs: &mut SparseObservable) -> *mut BitTerm {
    &mut obs.bit_terms_mut()[0]
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

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn obstermview_print(view: &CSparseTerm) {
    let rust_view: SparseTermView = view.try_into().unwrap();
    println!("{:?}", rust_view);
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bitterm_label(bit: &BitTerm) -> u32 {
    bit.py_label()
        .chars()
        .next()
        .expect("Label has exactly one character") as u32
}
