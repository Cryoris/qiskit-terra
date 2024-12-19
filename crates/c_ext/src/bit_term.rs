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

use qiskit_accelerate::sparse_observable::BitTerm;

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_term(bit: u8) -> *mut BitTerm {
    let term = BitTerm::try_from(bit).unwrap();
    Box::into_raw(Box::new(term))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_term_deallocate(bit: &mut BitTerm) {
    unsafe {
        let _ = Box::from_raw(bit);
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn bit_term_print(bit: &mut BitTerm) {
    println!("{:?}", bit);
}
