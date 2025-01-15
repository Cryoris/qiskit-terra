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

use qiskit_accelerate::sparse_observable::ArithmeticError;

use crate::sparse_observable::CInputError;

/// Integer exit codes returned to C.
#[repr(u32)]
pub enum ExitCode {
    Success = 0, // these need to be fixed for backward compat
    AlignmentError = 1,
    NullPointerError = 2,
    ArithmeticError = 3,
    IndexError = 4,
}

impl From<ArithmeticError> for ExitCode {
    fn from(_value: ArithmeticError) -> Self {
        ExitCode::ArithmeticError // do we want to cover each error enum here?
    }
}

impl From<CInputError> for ExitCode {
    fn from(value: CInputError) -> Self {
        match value {
            CInputError::AlignmentError => ExitCode::AlignmentError,
            CInputError::NullPointerError => ExitCode::NullPointerError,
        }
    }
}
