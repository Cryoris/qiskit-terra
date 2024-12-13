// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array2;
use num_complex::Complex64;
use num_traits::Zero;
use numpy::{
    PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{IntoPyDict, PyList, PyType},
    PyErr,
};
use qiskit_circuit::{
    imports::{ImportOnceCell, NUMPY_COPY_ONLY_IF_NEEDED},
    slice::{PySequenceIndex, SequenceIndex},
};
use std::{
    collections::btree_map,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
    sync::{Arc, RwLock},
};
use thiserror::Error;

use crate::sparse_observable::{
    ArithmeticError, BitTerm, BitTermFromU8Error, CoherenceError, LabelError, SparseObservable,
    SparseTerm, SparseTermView,
};

static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
static PAULI_LIST_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "PauliList");
static SPARSE_PAULI_OP_TYPE: ImportOnceCell =
    ImportOnceCell::new("qiskit.quantum_info", "SparsePauliOp");

static BIT_TERM_PY_ENUM: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static BIT_TERM_INTO_PY: GILOnceCell<[Option<Py<PyAny>>; 16]> = GILOnceCell::new();

#[derive(Error, Debug)]
struct InnerReadError;

#[derive(Error, Debug)]
struct InnerWriteError;

impl ::std::fmt::Display for InnerReadError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for reading.")
    }
}

impl ::std::fmt::Display for InnerWriteError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for writing.")
    }
}

impl From<InnerReadError> for PyErr {
    fn from(value: InnerReadError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}
impl From<InnerWriteError> for PyErr {
    fn from(value: InnerWriteError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}

impl From<BitTermFromU8Error> for PyErr {
    fn from(value: BitTermFromU8Error) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<CoherenceError> for PyErr {
    fn from(value: CoherenceError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<LabelError> for PyErr {
    fn from(value: LabelError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}
impl From<ArithmeticError> for PyErr {
    fn from(value: ArithmeticError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
impl IntoPy<Py<PyAny>> for BitTerm {
    fn into_py(self, py: Python) -> Py<PyAny> {
        let terms = BIT_TERM_INTO_PY.get_or_init(py, || {
            let py_enum = BIT_TERM_PY_ENUM
                .get_or_try_init(py, || make_py_bit_term(py))
                .expect("creating a simple Python enum class should be infallible")
                .bind(py);
            ::std::array::from_fn(|val| {
                ::bytemuck::checked::try_cast(val as u8)
                    .ok()
                    .map(|term: BitTerm| {
                        py_enum
                            .getattr(term.py_name())
                            .expect("the created `BitTerm` enum should have matching attribute names to the terms")
                            .unbind()
                    })
            })
        });
        terms[self as usize]
            .as_ref()
            .expect("the lookup table initializer populated a 'Some' in all valid locations")
            .clone_ref(py)
    }
}
impl ToPyObject for BitTerm {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        self.into_py(py)
    }
}
impl<'py> FromPyObject<'py> for BitTerm {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = ob
            .extract::<isize>()
            .map_err(|_| match ob.get_type().repr() {
                Ok(repr) => PyTypeError::new_err(format!("bad type for 'BitTerm': {}", repr)),
                Err(err) => err,
            })?;
        let value_error = || {
            PyValueError::new_err(format!(
                "value {} is not a valid letter of the single-qubit alphabet for 'BitTerm'",
                value
            ))
        };
        let value: u8 = value.try_into().map_err(|_| value_error())?;
        value.try_into().map_err(|_| value_error())
    }
}

/// A single term from a complete :class:`SparseObservable`.
///
/// These are typically created by indexing into or iterating through a :class:`SparseObservable`.
#[pyclass(name = "Term", frozen, module = "qiskit.quantum_info")]
#[derive(Clone, Debug)]
struct PySparseTerm {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    // Note that we can use the standard clone, since the PySparseTerm is immutable from Python
    // space, hence we can just clone the pointer. Otherwise we'd have to impl Clone manually,
    // like for the PySparseObservable.
    inner: Arc<RwLock<SparseTerm>>,
}

#[pymethods]
impl PySparseTerm {
    // Mark the Python class as being defined "within" the `SparseObservable` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "SparseObservable.Term"
    }

    #[new]
    #[pyo3(signature = (/, num_qubits, coeff, bit_terms, indices))]
    fn py_new(
        num_qubits: u32,
        coeff: Complex64,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
    ) -> PyResult<Self> {
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..bit_terms.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let bit_terms = order.iter().map(|i| bit_terms[*i]).collect();
        let mut sorted_indices = Vec::<u32>::with_capacity(order.len());
        for i in order {
            let index = indices[i];
            if sorted_indices
                .last()
                .map(|prev| *prev >= index)
                .unwrap_or(false)
            {
                return Err(CoherenceError::UnsortedIndices.into());
            }
            sorted_indices.push(index)
        }
        let inner = SparseTerm::new(
            num_qubits,
            coeff,
            bit_terms,
            sorted_indices.into_boxed_slice(),
        );
        Ok(PySparseTerm {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Convert this term to a complete :class:`SparseObservable`.
    fn to_observable(&self) -> PyResult<PySparseObservable> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let obs = SparseObservable::new(
            inner.num_qubits,
            vec![inner.coeff],
            inner.bit_terms.to_vec(),
            inner.indices.to_vec(),
            vec![0, inner.bit_terms.len()],
        )?;
        Ok(PySparseObservable {
            inner: Arc::new(RwLock::new(obs)),
        })
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.downcast_into::<Self>() else {
            return Ok(false);
        };
        let slf = slf.borrow();
        let other = other.borrow();
        let slf_inner = slf.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.eq(&other_inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(format!(
            "<{} on {} qubit{}: {}>",
            Self::type_qualname(),
            inner.num_qubits,
            if inner.num_qubits == 1 { "" } else { "s" },
            inner.view().to_sparse_str(),
        ))
    }

    fn __getnewargs__(slf_: Bound<Self>, py: Python) -> Py<PyAny> {
        let borrowed = slf_.borrow();
        let inner = borrowed.inner.read().unwrap();
        (
            inner.num_qubits,
            inner.coeff,
            Self::get_bit_terms(slf_.clone()),
            Self::get_indices(slf_),
        )
            .into_py(py)
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~SparseObservable.BitTerm`.
    #[getter]
    fn get_bit_terms(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let inner = borrowed.inner.read().unwrap();
        let bit_terms = &inner.bit_terms;
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(bit_terms));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[BitTerm]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array_bound(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// The number of qubits the term is defined on.
    #[getter]
    fn get_num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits)
    }

    /// The term's coefficient.
    #[getter]
    fn get_coeff(&self) -> PyResult<Complex64> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.coeff)
    }

    /// Read-only view onto the indices of each non-identity single-qubit term.
    ///
    /// The indices will always be in sorted order.
    #[getter]
    fn get_indices(slf_: Bound<Self>) -> Bound<PyArray1<u32>> {
        let borrowed = slf_.borrow();
        let inner = borrowed.inner.read().unwrap();
        let indices = &inner.indices;
        let arr = ::ndarray::aview1(indices);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array_bound(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// Get a :class:`.Pauli` object that represents the measurement basis needed for this term.
    ///
    /// For example, the projector ``0l+`` will return a Pauli ``ZXY``.  The resulting
    /// :class:`.Pauli` is dense, in the sense that explicit identities are stored.  An identity in
    /// the Pauli output does not require a concrete measurement.
    ///
    /// Returns:
    ///     :class:`.Pauli`: the Pauli operator representing the necessary measurement basis.
    ///
    /// See also:
    ///     :meth:`SparseObservable.pauli_bases`
    ///         A similar method for an entire observable at once.
    fn pauli_base<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mut x = vec![false; inner.num_qubits as usize];
        let mut z = vec![false; inner.num_qubits as usize];
        for (bit_term, index) in inner.bit_terms.iter().zip(inner.indices.iter()) {
            x[*index as usize] = bit_term.has_x_component();
            z[*index as usize] = bit_term.has_z_component();
        }
        PAULI_TYPE.get_bound(py).call1(((
            PyArray1::from_vec_bound(py, z),
            PyArray1::from_vec_bound(py, x),
        ),))
    }
}

/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `BitTerm`.
///
/// We don't make `BitTerm` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `SparseObservable` as a class attribute, and its
/// `__qualname__` is set to reflect this.
fn make_py_bit_term(py: Python) -> PyResult<Py<PyType>> {
    let terms = [
        BitTerm::X,
        BitTerm::Plus,
        BitTerm::Minus,
        BitTerm::Y,
        BitTerm::Right,
        BitTerm::Left,
        BitTerm::Z,
        BitTerm::Zero,
        BitTerm::One,
    ]
    .into_iter()
    .flat_map(|term| {
        let mut out = vec![(term.py_name(), term as u8)];
        if term.py_name() != term.py_label() {
            // Also ensure that the labels are created as aliases.  These can't be (easily) accessed
            // by attribute-getter (dot) syntax, but will work with the item-getter (square-bracket)
            // syntax, or programmatically with `getattr`.
            out.push((term.py_label(), term as u8));
        }
        out
    })
    .collect::<Vec<_>>();
    let obj = py.import_bound("enum")?.getattr("IntEnum")?.call(
        ("BitTerm", terms),
        Some(
            &[
                ("module", "qiskit.quantum_info"),
                ("qualname", "SparseObservable.BitTerm"),
            ]
            .into_py_dict_bound(py),
        ),
    )?;
    Ok(obj.downcast_into::<PyType>()?.unbind())
}

#[pyclass(name = "SparseObservable", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PySparseObservable {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    inner: Arc<RwLock<SparseObservable>>,
}

impl Clone for PySparseObservable {
    // we need to implement clone specifically to ensure the underlying inner data
    // is cloned, not the reference to it
    fn clone(&self) -> Self {
        let inner = self.inner.read().unwrap();
        Self {
            inner: Arc::new(RwLock::new(inner.clone())),
        }
    }
}

#[pymethods]
impl PySparseObservable {
    #[pyo3(signature = (data, /, num_qubits=None))]
    #[new]
    fn py_new(data: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = data.py();
        let check_num_qubits = |data: &Bound<PyAny>| -> PyResult<()> {
            let Some(num_qubits) = num_qubits else {
                return Ok(());
            };
            let other_qubits = data.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
            if num_qubits == other_qubits {
                return Ok(());
            }
            Err(PyValueError::new_err(format!(
                "explicitly given 'num_qubits' ({num_qubits}) does not match operator ({other_qubits})"
            )))
        };

        if data.is_instance(PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
        }
        if data.is_instance(SPARSE_PAULI_OP_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_sparse_pauli_op(data);
        }
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label).map_err(PyErr::from);
        }
        if let Ok(observable) = data.downcast_exact::<Self>() {
            check_num_qubits(data)?;
            return Ok(observable.borrow().clone());
        }
        // The type of `vec` is inferred from the subsequent calls to `Self::py_from_list` or
        // `Self::py_from_sparse_list` to be either the two-tuple or the three-tuple form during the
        // `extract`.  The empty list will pass either, but it means the same to both functions.
        if let Ok(vec) = data.extract() {
            return Self::from_list(vec, num_qubits);
        }
        if let Ok(vec) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-list form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_list(vec, num_qubits).map_err(PyErr::from);
        }
        if let Ok(term) = data.downcast_exact::<PySparseTerm>() {
            return term.borrow().to_observable();
        };
        if let Ok(observable) = Self::from_terms(data, num_qubits) {
            return Ok(observable);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'SparseObservable': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this observable.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = SparseObservable.from_list([("IXZ+lr01", 2.5), ("ZXI-rl10", 0.5j)])
    ///         >>> assert obs == obs.copy()
    ///         >>> assert obs is not obs.copy()
    fn copy(&self) -> Self {
        self.clone()
    }

    /// The number of qubits the operator acts on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of terms in the sum this operator is tracking.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// The coefficients of each abstract term in in the sum.  This has as many elements as terms in
    /// the sum.
    #[getter]
    fn get_coeffs(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Coeffs,
        }
    }

    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    #[getter]
    fn get_bit_terms(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::BitTerms,
        }
    }

    /// A flat list of the qubit indices that the corresponding entries in :attr:`bit_terms` act on.
    /// This list must always be term-wise sorted, where a term is a sublist as denoted by
    /// :attr:`boundaries`.
    ///
    /// .. warning::
    ///
    ///     If writing to this attribute from Python space, you *must* ensure that you only write in
    ///     indices that are term-wise sorted.
    #[getter]
    fn get_indices(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Indices,
        }
    }

    /// Indices that partition :attr:`bit_terms` and :attr:`indices` into sublists for each
    /// individual term in the sum.  ``boundaries[0] : boundaries[1]`` is the range of indices into
    /// :attr:`bit_terms` and :attr:`indices` that correspond to the first term of the sum.  All
    /// unspecified qubit indices are implicitly the identity.  This is one item longer than
    /// :attr:`coeffs`, since ``boundaries[0]`` is always an explicit zero (for algorithmic ease).
    #[getter]
    fn get_boundaries(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Boundaries,
        }
    }

    /// Get the zero operator over the given number of qubits.
    ///
    /// The zero operator is the operator whose expectation value is zero for all quantum states.
    /// It has no terms.  It is the identity element for addition of two :class:`SparseObservable`
    /// instances; anything added to the zero operator is equal to itself.
    ///
    /// If you want the projector onto the all zeros state, use::
    ///
    ///     >>> num_qubits = 10
    ///     >>> all_zeros = SparseObservable.from_label("0" * num_qubits)
    ///
    /// Examples:
    ///
    ///     Get the zero operator for 100 qubits::
    ///
    ///         >>> SparseObservable.zero(100)
    ///         <SparseObservable with 0 terms on 100 qubits: 0.0>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn zero(num_qubits: u32) -> Self {
        let inner = SparseObservable::zero(num_qubits);
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Get the identity operator over the given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity operator for 100 qubits::
    ///
    ///         >>> SparseObservable.identity(100)
    ///         <SparseObservable with 1 term on 100 qubits: (1+0j)()>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        let inner = SparseObservable::identity(num_qubits);
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Construct a :class:`.SparseObservable` from a single :class:`.Pauli` instance.
    ///
    /// The output observable will have a single term, with a unitary coefficient dependent on the
    /// phase.
    ///
    /// Args:
    ///     pauli (:class:`.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///         
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> SparseObservable.from_pauli(pauli)
    ///         <SparseObservable with 1 term on 5 qubits: (1+0j)(Y_3 X_2 Z_1)>
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut num_ys = 0;
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast((*x as u8) << 1 | (*z as u8)) else {
                continue;
            };
            num_ys += (term == BitTerm::Y) as isize;
            indices.push(i as u32);
            bit_terms.push(term);
        }
        let boundaries = vec![0, indices.len()];
        // The "empty" state of a `Pauli` represents the identity, which isn't our empty state
        // (that's zero), so we're always going to have a coefficient.
        let group_phase = pauli
            // `Pauli`'s `_phase` is a Numpy array ...
            .getattr(intern!(py, "_phase"))?
            // ... that should have exactly 1 element ...
            .call_method0(intern!(py, "item"))?
            // ... which is some integral type.
            .extract::<isize>()?;
        let phase = match (group_phase - num_ys).rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, -1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, 1.0),
            _ => unreachable!("`x % 4` has only four values"),
        };
        let coeffs = vec![phase];
        let inner = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Construct a single-term observable from a dense string label.
    ///
    /// The resulting operator will have a coefficient of 1.  The label must be a sequence of the
    /// alphabet ``'IXYZ+-rl01'``.  The label is interpreted analogously to a bitstring.  In other
    /// words, the right-most letter is associated with qubit 0, and so on.  This is the same as the
    /// labels for :class:`.Pauli` and :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///         
    ///         >>> SparseObservable.from_label("IIII+ZI")
    ///         <SparseObservable with 1 term on 7 qubits: (1+0j)(+_2 Z_1)>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a sum operator from multiple labels
    ///         and their corresponding coefficients.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> Result<Self, LabelError> {
        let mut inner = SparseObservable::zero(label.len() as u32);
        inner.add_dense_label(label, Complex64::new(1.0, 0.0))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Construct an observable from a list of dense labels and coefficients.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.SparseObservable`.  In
    /// this dense form, you must supply all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ+-rl01'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[tuple[str, complex]]): Pairs of labels and their associated coefficients to
    ///         sum. The labels are interpreted the same way as in :meth:`from_label`.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the observable is for.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct an observable from a list of labels of the same length::
    ///
    ///         >>> SparseObservable.from_list([
    ///         ...     ("III++", 1.0),
    ///         ...     ("II--I", 1.0j),
    ///         ...     ("I++II", -0.5),
    ///         ...     ("--III", -0.25j),
    ///         ... ])
    ///         <SparseObservable with 4 terms on 5 qubits:
    ///             (1+0j)(+_1 +_0) + (0+1j)(-_2 -_1) + (-0.5+0j)(+_3 +_2) + (-0-0.25j)(-_4 -_3)>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> SparseObservable.from_list([], num_qubits=10)
    ///         <SparseObservable with 0 terms on 10 qubits: 0.0>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = SparseObservable.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = SparseObservable.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_label`
    ///         A similar constructor, but takes only a single label and always has its coefficient
    ///         set to ``1.0``.
    ///
    ///     :meth:`from_sparse_list`
    ///         Construct the observable from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<(String, Complex64)>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct an observable from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].0.len() as u32,
        };
        let mut inner = SparseObservable::with_capacity(num_qubits, iter.len(), 0);
        for (label, coeff) in iter {
            inner.add_dense_label(&label, coeff)?;
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Construct an observable from a list of labels, the qubits each item applies to, and the
    /// coefficient of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.SparseObservable`.
    ///
    /// The "labels" and "indices" fields of the triples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int], complex]]): triples of labels, the qubits
    ///         each single-qubit term applies to, and the coefficient of the entire term.
    ///
    ///     num_qubits (int): the number of qubits in the operator.
    ///
    /// Examples:
    ///
    ///     Construct a simple operator::
    ///
    ///         >>> SparseObservable.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2j)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <SparseObservable with 2 terms on 5 qubits: (1+0j)(X_4 Z_1) + (0+2j)(Y_3 Y_0)>
    ///
    ///     Construct the identity observable (though really, just use :meth:`identity`)::
    ///
    ///         >>> SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=100)
    ///         <SparseObservable with 1 term on 100 qubits: (1+0j)()>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = SparseObservable.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = SparseObservable.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(
        iter: Vec<(String, Vec<u32>, Complex64)>,
        num_qubits: u32,
    ) -> PyResult<Self> {
        let coeffs = iter.iter().map(|(_, _, coeff)| *coeff).collect();
        let mut boundaries = Vec::with_capacity(iter.len() + 1);
        boundaries.push(0);
        let mut indices = Vec::new();
        let mut bit_terms = Vec::new();
        // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
        // sorting on-the-fly.
        let mut sorted = btree_map::BTreeMap::new();
        for (label, qubits, _) in iter {
            sorted.clear();
            let label: &[u8] = label.as_ref();
            if label.len() != qubits.len() {
                return Err(LabelError::WrongLengthIndices {
                    label: label.len(),
                    indices: indices.len(),
                }
                .into());
            }
            for (letter, index) in label.iter().zip(qubits) {
                if index >= num_qubits {
                    return Err(LabelError::BadIndex { index, num_qubits }.into());
                }
                let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                    return Err(LabelError::DuplicateIndex { index }.into());
                };
                entry.insert(
                    BitTerm::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?,
                );
            }
            for (index, term) in sorted.iter() {
                let Some(term) = term else {
                    continue;
                };
                indices.push(*index);
                bit_terms.push(*term);
            }
            boundaries.push(bit_terms.len());
        }
        let inner = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Construct a :class:`.SparseObservable` from a :class:`.SparsePauliOp` instance.
    ///
    /// This will be a largely direct translation of the :class:`.SparsePauliOp`; in particular,
    /// there is no on-the-fly summing of like terms, nor any attempt to refactorize sums of Pauli
    /// terms into equivalent projection operators.
    ///
    /// Args:
    ///     op (:class:`.SparsePauliOp`): the operator to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> spo = SparsePauliOp.from_list([("III", 1.0), ("IIZ", 0.5), ("IZI", 0.5)])
    ///         >>> SparseObservable.from_sparse_pauli_op(spo)
    ///         <SparseObservable with 3 terms on 3 qubits: (1+0j)() + (0.5+0j)(Z_0) + (0.5+0j)(Z_1)>
    #[staticmethod]
    #[pyo3(signature = (op, /))]
    fn from_sparse_pauli_op(op: &Bound<PyAny>) -> PyResult<Self> {
        let py = op.py();
        let pauli_list_ob = op.getattr(intern!(py, "paulis"))?;
        let coeffs = op
            .getattr(intern!(py, "coeffs"))?
            .extract::<PyReadonlyArray1<Complex64>>()
            .map_err(|_| PyTypeError::new_err("only 'SparsePauliOp' with complex-typed coefficients can be converted to 'SparseObservable'"))?
            .as_array()
            .to_vec();
        let op_z = pauli_list_ob
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        let op_x = pauli_list_ob
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        // We don't extract the `phase`, because that's supposed to be 0 for all `SparsePauliOp`
        // instances - they use the symplectic convention in the representation with any phase term
        // absorbed into the coefficients (like us).
        let [num_terms, num_qubits] = *op_z.shape() else {
            unreachable!("shape is statically known to be 2D")
        };
        if op_x.shape() != [num_terms, num_qubits] {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have different shapes ({:?} and {:?})",
                op_x.shape(),
                op_z.shape()
            )));
        }
        if num_terms != coeffs.len() {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have a different number of operators to 'coeffs' ({} and {})",
                num_terms,
                coeffs.len(),
            )));
        }

        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut boundaries = Vec::with_capacity(num_terms + 1);
        boundaries.push(0);
        for (term_x, term_z) in op_x
            .as_array()
            .rows()
            .into_iter()
            .zip(op_z.as_array().rows())
        {
            for (i, (x, z)) in term_x.iter().zip(term_z.iter()).enumerate() {
                // The only failure case possible here is the identity, because of how we're
                // constructing the value to convert.
                let Ok(term) = ::bytemuck::checked::try_cast((*x as u8) << 1 | (*z as u8)) else {
                    continue;
                };
                indices.push(i as u32);
                bit_terms.push(term);
            }
            boundaries.push(indices.len());
        }

        let inner =
            SparseObservable::new(num_qubits as u32, coeffs, bit_terms, indices, boundaries)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Construct a :class:`SparseObservable` out of individual terms.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// No simplification is done as part of the observable creation.
    ///
    /// Args:
    ///     obj (Iterable[Term]): Iterable of individual terms to build the observable from.
    ///     num_qubits (int | None): The number of qubits the observable should act on.  This is
    ///         usually inferred from the input, but can be explicitly given to handle the case
    ///         of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding observable.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_terms(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => SparseObservable::zero(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an observable from an empty list without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.downcast::<PySparseTerm>()?.borrow();
                let term = py_term.inner.read().map_err(|_| InnerReadError)?;
                term.to_observable()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.downcast::<PySparseTerm>()?.borrow();
            let term = py_term.inner.read().map_err(|_| InnerReadError)?;
            inner.add_term(term.view())?;
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.SparseObservable` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation <sparse-observable-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits in the observable.
    ///     coeffs: complex coefficients of each term of the observable.  This should be a Numpy
    ///         array with dtype :attr:`~numpy.complex128`.
    ///     bit_terms: flattened list of the single-qubit terms comprising all complete terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uint8` (which is compatible with
    ///         :class:`.BitTerm`).
    ///     indices: flattened term-wise sorted list of the qubits each single-qubit term corresponds
    ///         to.  This should be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///     boundaries: the indices that partition ``bit_terms`` and ``indices`` into terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uintp`.
    ///     check: if ``True`` (the default), validate that the data satisfies all coherence
    ///         guarantees.  If ``False``, no checks are done.
    ///
    ///         .. warning::
    ///
    ///             If ``check=False``, the ``bit_terms`` absolutely *must* be all be valid values
    ///             of :class:`.SparseObservable.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), SparseObservable.BitTerm.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> coeffs = np.ones((num_qubits,), dtype=complex)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> SparseObservable.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
    ///         <SparseObservable with 100 terms on 100 qubits: (1+0j)(Z_0) + ... + (1+0j)(Z_99)>
    #[deny(unsafe_op_in_unsafe_fn)]
    #[staticmethod]
    #[pyo3(
        signature = (/, num_qubits, coeffs, bit_terms, indices, boundaries, check=true),
    )]
    unsafe fn from_raw_parts<'py>(
        num_qubits: u32,
        coeffs: PyArrayLike1<'py, Complex64>,
        bit_terms: PyArrayLike1<'py, u8>,
        indices: PyArrayLike1<'py, u32>,
        boundaries: PyArrayLike1<'py, usize>,
        check: bool,
    ) -> PyResult<Self> {
        let coeffs = coeffs.as_array().to_vec();
        let bit_terms = if check {
            bit_terms
                .as_array()
                .into_iter()
                .copied()
                .map(BitTerm::try_from)
                .collect::<Result<_, _>>()?
        } else {
            let bit_terms_as_u8 = bit_terms.as_array().to_vec();
            // SAFETY: the caller enforced that each `u8` is a valid `BitTerm`, and `BitTerm` is be
            // represented by a `u8`.  We can't use `bytemuck` because we're casting a `Vec`.
            unsafe { ::std::mem::transmute::<Vec<u8>, Vec<BitTerm>>(bit_terms_as_u8) }
        };
        let indices = indices.as_array().to_vec();
        let boundaries = boundaries.as_array().to_vec();

        let inner = if check {
            SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
                .map_err(PyErr::from)
        } else {
            // SAFETY: the caller promised they have upheld the coherence guarantees.
            Ok(unsafe {
                SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries)
            })
        }?;

        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// substraction operations may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = SparseObservable.from_list([("IX+-rl", 2.0), ("01YZII", -1j)])
    ///         >>> obs.clear()
    ///         >>> assert obs == SparseObservable.zero(obs.py_num_qubits())
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Sum any like terms in this operator, removing them if the resulting complex coefficient has
    /// an absolute value within tolerance of zero.
    ///
    /// As a side effect, this sorts the operator into :ref:`canonical order
    /// <sparse-observable-canonical-order>`.
    ///
    /// .. note::
    ///
    ///     When using this for equality comparisons, note that floating-point rounding and the
    ///     non-associativity fo floating-point addition may cause non-zero coefficients of summed
    ///     terms to compare unequal.  To compare two observables up to a tolerance, it is safest to
    ///     compare the canonicalized difference of the two observables to zero.
    ///
    /// Args:
    ///     tol (float): after summing like terms, any coefficients whose absolute value is less
    ///         than the given absolute tolerance will be suppressed from the output.
    ///
    /// Examples:
    ///
    ///     Using :meth:`simplify` to compare two operators that represent the same observable, but
    ///     would compare unequal due to the structural tests by default::
    ///
    ///         >>> base = SparseObservable.from_sparse_list([
    ///         ...     ("XZ", (2, 1), 1e-10),  # value too small
    ///         ...     ("+-", (3, 1), 2j),
    ///         ...     ("+-", (3, 1), 2j),     # can be combined with the above
    ///         ...     ("01", (3, 1), 0.5),    # out of order compared to `expected`
    ///         ... ], num_qubits=5)
    ///         >>> expected = SparseObservable.from_list([("I0I1I", 0.5), ("I+I-I", 4j)])
    ///         >>> assert base != expected  # non-canonical comparison
    ///         >>> assert base.simplify() == expected.simplify()
    ///
    ///     Note that in the above example, the coefficients are chosen such that all floating-point
    ///     calculations are exact, and there are no intermediate rounding or associativity
    ///     concerns.  If this cannot be guaranteed to be the case, the safer form is::
    ///
    ///         >>> left = SparseObservable.from_list([("XYZ", 1.0/3.0)] * 3)   # sums to 1.0
    ///         >>> right = SparseObservable.from_list([("XYZ", 1.0/7.0)] * 7)  # doesn't sum to 1.0
    ///         >>> assert left.simplify() != right.simplify()
    ///         >>> assert (left - right).simplify() == SparseObservable.zero(left.num_qubits)
    #[pyo3(
        signature = (/, tol=1e-8),
    )]
    fn simplify(&self, tol: f64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let simplified = inner.canonicalize(tol);
        Ok(Self {
            inner: Arc::new(RwLock::new(simplified)),
        })
    }

    /// Calculate the adjoint of this observable.
    ///
    ///
    /// This is well defined in the abstract mathematical sense.  All the terms of the single-qubit
    /// alphabet are self-adjoint, so the result of this operation is the same observable, except
    /// its coefficients are all their complex conjugates.
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> left = SparseObservable.from_list([("XY+-", 1j)])
    ///         >>> right = SparseObservable.from_list([("XY+-", -1j)])
    ///         >>> assert left.adjoint() == right
    fn adjoint(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner.adjoint())),
        })
    }

    /// Calculate the matrix transposition of this observable.
    ///
    /// This operation is defined in terms of the standard matrix conventions of Qiskit, in that the
    /// matrix form is taken to be in the $Z$ computational basis.  The $X$- and $Z$-related
    /// alphabet terms are unaffected by the transposition, but $Y$-related terms modify their
    /// alphabet terms.  Precisely:
    ///
    /// * :math:`Y` transposes to :math:`-Y`
    /// * :math:`\lvert r\rangle\langle r\rvert` transposes to :math:`\lvert l\rangle\langle l\rvert`
    /// * :math:`\lvert l\rangle\langle l\rvert` transposes to :math:`\lvert r\rangle\langle r\rvert`
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> obs = SparseObservable([("III", 1j), ("Yrl", 0.5)])
    ///         >>> assert obs.transpose() == SparseObservable([("III", 1j), ("Ylr", -0.5)])
    fn transpose(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner.transpose())),
        })
    }

    /// Calculate the complex conjugation of this observable.
    ///
    /// This operation is defined in terms of the standard matrix conventions of Qiskit, in that the
    /// matrix form is taken to be in the $Z$ computational basis.  The $X$- and $Z$-related
    /// alphabet terms are unaffected by the complex conjugation, but $Y$-related terms modify their
    /// alphabet terms.  Precisely:
    ///
    /// * :math:`Y` conjguates to :math:`-Y`
    /// * :math:`\lvert r\rangle\langle r\rvert` conjugates to :math:`\lvert l\rangle\langle l\rvert`
    /// * :math:`\lvert l\rangle\langle l\rvert` conjugates to :math:`\lvert r\rangle\langle r\rvert`
    ///
    /// Additionally, all coefficients are conjugated.
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> obs = SparseObservable([("III", 1j), ("Yrl", 0.5)])
    ///         >>> assert obs.conjugate() == SparseObservable([("III", -1j), ("Ylr", -0.5)])
    fn conjugate(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner.conjugate())),
        })
    }

    /// Tensor product of two observables.
    ///
    /// The bit ordering is defined such that the qubit indices of the argument will remain the
    /// same, and the indices of ``self`` will be offset by the number of qubits in ``other``.  This
    /// is the same convention as used by the rest of Qiskit's :mod:`~qiskit.quantum_info`
    /// operators.
    ///
    /// This function is used for the infix ``^`` operator.  If using this operator, beware that
    /// `Python's operator-precedence rules
    /// <https://docs.python.org/3/reference/expressions.html#operator-precedence>`__ may cause the
    /// evaluation order to be different to your expectation.  In particular, the operator ``+``
    /// binds more tightly than ``^``, just like ``*`` binds more tightly than ``+``.  Use
    /// parentheses to fix the evaluation order, if needed.
    ///
    /// The argument will be cast to :class:`SparseObservable` using its default constructor, if it
    /// is not already in the correct form.
    ///
    /// Args:
    ///
    ///     other: the observable to put on the right-hand side of the tensor product.
    ///
    /// Examples:
    ///
    ///     The bit ordering of this is such that the tensor product of two observables made from a
    ///     single label "looks like" an observable made by concatenating the two strings::
    ///
    ///         >>> left = SparseObservable.from_label("XYZ")
    ///         >>> right = SparseObservable.from_label("+-IIrl")
    ///         >>> assert left.tensor(right) == SparseObservable.from_label("XYZ+-IIrl")
    ///
    ///     You can also use the infix ``^`` operator for tensor products, which will similarly cast
    ///     the right-hand side of the operation if it is not already a :class:`SparseObservable`::
    ///
    ///         >>> assert SparseObservable("rl") ^ Pauli("XYZ") == SparseObservable("rlXYZ")
    ///
    /// See also:
    ///     :meth:`expand`
    ///
    ///         The same function, but with the order of arguments flipped.  This can be useful if
    ///         you like using the casting behavior for the argument, but you want your existing
    ///         :class:`SparseObservable` to be on the right-hand side of the tensor ordering.
    #[pyo3(signature = (other, /))]
    fn tensor(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for tensor: {}",
                other.get_type().repr()?
            )));
        };

        let other = other.borrow();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;

        let result = inner.tensor(&other_inner);
        Ok(Self {
            inner: Arc::new(RwLock::new(result)),
        }
        .into_py(py))
    }

    /// Reverse-order tensor product.
    ///
    /// This is equivalent to ``other.tensor(self)``, except that ``other`` will first be type-cast
    /// to :class:`SparseObservable` if it isn't already one (by calling the default constructor).
    ///
    /// Args:
    ///
    ///     other: the observable to put on the left-hand side of the tensor product.
    ///
    /// Examples:
    ///
    ///     This is equivalent to :meth:`tensor` with the order of the arguments flipped::
    ///
    ///         >>> left = SparseObservable.from_label("XYZ")
    ///         >>> right = SparseObservable.from_label("+-IIrl")
    ///         >>> assert left.tensor(right) == right.expand(left)
    ///
    /// See also:
    ///     :meth:`tensor`
    ///
    ///         The same function with the order of arguments flipped.  :meth:`tensor` is the more
    ///         standard argument ordering, and matches Qiskit's other conventions.
    #[pyo3(signature = (other, /))]
    fn expand(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for expand: {}",
                other.get_type().repr()?
            )));
        };

        let other = other.borrow();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;

        let result = other_inner.tensor(&inner);
        Ok(Self {
            inner: Arc::new(RwLock::new(result)),
        }
        .into_py(py))
    }

    /// Apply a transpiler layout to this :class:`SparseObservable`.
    ///
    /// Typically you will have defined your observable in terms of the virtual qubits of the
    /// circuits you will use to prepare states.  After transpilation, the virtual qubits are mapped
    /// to particular physical qubits on a device, which may be wider than your circuit.  That
    /// mapping can also change over the course of the circuit.  This method transforms the input
    /// observable on virtual qubits to an observable that is suitable to apply immediately after
    /// the fully transpiled *physical* circuit.
    ///
    /// Args:
    ///     layout (TranspileLayout | list[int] | None): The layout to apply.  Most uses of this
    ///         function should pass the :attr:`.QuantumCircuit.layout` field from a circuit that
    ///         was transpiled for hardware.  In addition, you can pass a list of new qubit indices.
    ///         If given as explicitly ``None``, no remapping is applied (but you can still use
    ///         ``num_qubits`` to expand the observable).
    ///     num_qubits (int | None): The number of qubits to expand the observable to.  If not
    ///         supplied, the output will be as wide as the given :class:`.TranspileLayout`, or the
    ///         same width as the input if the ``layout`` is given in another form.
    ///
    /// Returns:
    ///     A new :class:`SparseObservable` with the provided layout applied.
    #[pyo3(signature = (/, layout, num_qubits=None))]
    fn apply_layout(&self, layout: Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = layout.py();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let check_inferred_qubits = |inferred: u32| -> PyResult<u32> {
            if inferred < inner.num_qubits() {
                return Err(CoherenceError::NotEnoughQubits {
                    current: inner.num_qubits() as usize,
                    target: inferred as usize,
                }
                .into());
            }
            Ok(inferred)
        };

        // get the number of qubits in the layout and map the layout to Vec<u32> to
        // call SparseObservable.apply_layout
        let (num_qubits, layout): (u32, Option<Vec<u32>>) = if layout.is_none() {
            // if the layout is none,
            (num_qubits.unwrap_or(inner.num_qubits()), None)
        } else {
            if layout.is_instance(
                &py.import_bound(intern!(py, "qiskit.transpiler"))?
                    .getattr(intern!(py, "TranspileLayout"))?,
            )? {
                (
                    check_inferred_qubits(
                        layout.getattr(intern!(py, "_output_qubit_list"))?.len()? as u32,
                    )?,
                    Some(
                        layout
                            .call_method0(intern!(py, "final_index_layout"))?
                            .extract::<Vec<u32>>()?,
                    ),
                )
            } else {
                (
                    check_inferred_qubits(num_qubits.unwrap_or(inner.num_qubits()))?,
                    Some(layout.extract()?),
                )
            }
        };
        let out = inner.apply_layout(layout.as_deref(), num_qubits)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(out)),
        })
    }

    /// Get a :class:`.PauliList` object that represents the measurement basis needed for each term
    /// (in order) in this observable.
    ///
    /// For example, the projector ``0l+`` will return a Pauli ``ZXY``.  The resulting
    /// :class:`.Pauli` is dense, in the sense that explicit identities are stored.  An identity in
    /// the Pauli output does not require a concrete measurement.
    ///
    /// This will return an entry in the Pauli list for every term in the sum.
    ///
    /// Returns:
    ///     :class:`.PauliList`: the Pauli operator list representing the necessary measurement
    ///     bases.
    fn pauli_bases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mut x = Array2::from_elem([inner.num_terms(), inner.num_qubits() as usize], false);
        let mut z = Array2::from_elem([inner.num_terms(), inner.num_qubits() as usize], false);
        for (loc, term) in inner.iter().enumerate() {
            let mut x_row = x.row_mut(loc);
            let mut z_row = z.row_mut(loc);
            for (bit_term, index) in term.bit_terms.iter().zip(term.indices) {
                x_row[*index as usize] = bit_term.has_x_component();
                z_row[*index as usize] = bit_term.has_z_component();
            }
        }
        PAULI_LIST_TYPE
            .get_bound(py)
            .getattr(intern!(py, "from_symplectic"))?
            .call1((
                PyArray2::from_owned_array_bound(py, z),
                PyArray2::from_owned_array_bound(py, x),
            ))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<Py<PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                let term = inner.term(index).to_term();
                let py_term = PySparseTerm {
                    inner: Arc::new(RwLock::new(term)),
                };
                return Ok(py_term.into_py(py));
            }
            indices => indices,
        };
        let mut out = SparseObservable::zero(inner.num_qubits());
        for index in indices.iter() {
            out.add_term(inner.term(index))?;
        }

        let py_out = Self {
            inner: Arc::new(RwLock::new(out)),
        };
        Ok(py_out.into_py(py))
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        // this is also important to check before trying to read both slf and other
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.downcast_into::<Self>() else {
            return Ok(false);
        };
        let slf_borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        let slf_inner = slf_borrowed.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other_borrowed.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.eq(&other_inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        let num_terms = self.num_terms()?;
        let num_qubits = self.num_qubits()?;

        let str_num_terms = format!(
            "{} term{}",
            num_terms,
            if num_terms == 1 { "" } else { "s" }
        );
        let str_num_qubits = format!(
            "{} qubit{}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" }
        );

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let str_terms = if num_terms == 0 {
            "0.0".to_owned()
        } else {
            inner
                .iter()
                .map(SparseTermView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(" + ")
        };
        Ok(format!(
            "<SparseObservable with {} on {}: {}>",
            str_num_terms, str_num_qubits, str_terms
        ))
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // we acquire the read lock once here and use the internal methods
        // to avoid checking and acquiring the lock in every method call
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let bit_terms: &[u8] = ::bytemuck::cast_slice(&inner.bit_terms());
        Ok((
            py.get_type_bound::<Self>().getattr("from_raw_parts")?,
            (
                inner.num_qubits(),
                PyArray1::from_slice_bound(py, inner.coeffs()),
                PyArray1::from_slice_bound(py, bit_terms),
                PyArray1::from_slice_bound(py, inner.indices()),
                PyArray1::from_slice_bound(py, inner.boundaries()),
                false,
            ),
        )
            .into_py(py))
    }

    fn __add__(slf_: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf_.py();
        if slf_.is(other) {
            // This fast path is for consistency with the in-place `__iadd__`, which would otherwise
            // struggle to do the addition to itself.
            let slf_ = slf_.borrow();
            let inner = slf_.inner.read().map_err(|_| InnerReadError)?;
            let doubled =
                <&SparseObservable as ::std::ops::Mul<_>>::mul(&inner, Complex64::new(2.0, 0.0));
            return Ok(Self {
                inner: Arc::new(RwLock::new(doubled)),
            }
            .into_py(py));
        }
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };
        let slf_ = slf_.borrow();
        let slf_inner = slf_.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        let added = <&SparseObservable as ::std::ops::Add>::add(&slf_inner, &other_inner);

        Ok(Self {
            inner: Arc::new(RwLock::new(added)),
        }
        .into_py(py))
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        // No need to handle the `self is other` case here, because `__add__` will get it.
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };
        // let other = other.borrow();

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        inner.check_equal_qubits(&other_inner)?;
        let added = <&SparseObservable as ::std::ops::Add>::add(&other_inner, &inner);

        Ok(Self {
            inner: Arc::new(RwLock::new(added)),
        }
        .into_py(py))
    }

    fn __iadd__(slf_: Bound<PySparseObservable>, other: &Bound<PyAny>) -> PyResult<()> {
        if slf_.is(other) {
            let slf_ = slf_.borrow();
            let mut slf_inner = slf_.inner.write().map_err(|_| InnerWriteError)?;
            *slf_inner *= Complex64::new(2.0, 0.0);
            return Ok(());
        }
        let Some(other) = coerce_to_observable(other)? else {
            // This is not well behaved - we _should_ return `NotImplemented` to Python space
            // without an exception, but limitations in PyO3 prevent this at the moment.  See
            // https://github.com/PyO3/pyo3/issues/4605.
            return Err(PyTypeError::new_err(format!(
                "invalid object for in-place addition of 'SparseObservable': {}",
                other.repr()?
            )));
        };
        let slf_ = slf_.borrow();
        let mut slf_inner = slf_.inner.write().map_err(|_| InnerWriteError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        slf_inner.add_assign(&other_inner);
        Ok(())
    }

    fn __sub__(slf_: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf_.py();
        if slf_.is(other) {
            return Ok(PySparseObservable::zero(slf_.borrow().num_qubits()?).into_py(py));
        }
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };

        let slf_ = slf_.borrow();
        let slf_inner = slf_.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        let subtracted = <&SparseObservable as ::std::ops::Sub>::sub(&slf_inner, &other_inner);

        Ok(Self {
            inner: Arc::new(RwLock::new(subtracted)),
        }
        .into_py(py))
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        inner.check_equal_qubits(&other_inner)?;
        let subtracted = <&SparseObservable as ::std::ops::Sub>::sub(&other_inner, &inner);

        Ok(Self {
            inner: Arc::new(RwLock::new(subtracted)),
        }
        .into_py(py))
    }

    fn __isub__(slf_: Bound<PySparseObservable>, other: &Bound<PyAny>) -> PyResult<()> {
        if slf_.is(other) {
            // This is not strictly the same thing as `a - a` if `a` contains non-finite
            // floating-point values (`inf - inf` is `NaN`, for example); we don't really have a
            // clear view on what floating-point guarantees we're going to make right now.
            slf_.borrow_mut().clear()?;
            return Ok(());
        }
        let Some(other) = coerce_to_observable(other)? else {
            // This is not well behaved - we _should_ return `NotImplemented` to Python space
            // without an exception, but limitations in PyO3 prevent this at the moment.  See
            // https://github.com/PyO3/pyo3/issues/4605.
            return Err(PyTypeError::new_err(format!(
                "invalid object for in-place subtraction of 'SparseObservable': {}",
                other.repr()?
            )));
        };
        let slf_ = slf_.borrow();
        let mut slf_inner = slf_.inner.write().map_err(|_| InnerWriteError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        slf_inner.sub_assign(&other_inner);
        Ok(())
    }

    fn __pos__(&self) -> Self {
        self.clone()
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let neg = <&SparseObservable as ::std::ops::Neg>::neg(&inner);
        Ok(Self {
            inner: Arc::new(RwLock::new(neg)),
        })
    }

    fn __mul__(&self, other: Complex64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mult = <&SparseObservable as ::std::ops::Mul<_>>::mul(&inner, other);
        Ok(Self {
            inner: Arc::new(RwLock::new(mult)),
        })
    }
    fn __rmul__(&self, other: Complex64) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __imul__(&mut self, other: Complex64) -> PyResult<()> {
        // we obtain write access to ``inner`` and call SparseObservable::mul_assign to mutate the
        // the ``inner`` data directly
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.mul_assign(other);
        Ok(())
    }

    fn __truediv__(&self, other: Complex64) -> PyResult<Self> {
        if other.is_zero() {
            return Err(PyZeroDivisionError::new_err("complex division by zero"));
        }
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let div = <&SparseObservable as ::std::ops::Div<_>>::div(&inner, other);
        Ok(Self {
            inner: Arc::new(RwLock::new(div)),
        })
    }
    fn __itruediv__(&mut self, other: Complex64) -> PyResult<()> {
        if other.is_zero() {
            return Err(PyZeroDivisionError::new_err("complex division by zero"));
        }
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.div_assign(other);
        Ok(())
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        // we cannot just delegate this to ``tensor`` since ``other`` might allow
        // right-hand-side arithmetic and we have to try deferring to that object,
        // which is done by returning ``NotImplemented``
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };

        self.tensor(&other)
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };

        self.expand(&other)
    }

    // The documentation for this is inlined into the class-level documentation of
    // `SparseObservable`.
    #[allow(non_snake_case)]
    #[classattr]
    fn BitTerm(py: Python) -> PyResult<Py<PyType>> {
        BIT_TERM_PY_ENUM
            .get_or_try_init(py, || make_py_bit_term(py))
            .map(|obj| obj.clone_ref(py))
    }

    // The documentation for this is inlined into the class-level documentation of
    // `SparseObservable`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Term(py: Python) -> Bound<PyType> {
        py.get_type_bound::<PySparseTerm>()
    }
}

/// Helper class of `ArrayView` that denotes the slot of the `SparseObservable` we're looking at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArraySlot {
    Coeffs,
    BitTerms,
    Indices,
    Boundaries,
}

/// Custom wrapper sequence class to get safe views onto the Rust-space data.  We can't directly
/// expose Python-managed wrapped pointers without introducing some form of runtime exclusion on the
/// ability of `SparseObservable` to re-allocate in place; we can't leave dangling pointers for
/// Python space.
#[pyclass(frozen, sequence)]
struct ArrayView {
    base: Py<PySparseObservable>,
    slot: ArraySlot,
}
#[pymethods]
impl ArrayView {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let borrowed = self.base.borrow(py);
        let obs = borrowed.inner.read().map_err(|_| InnerReadError)?;
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", obs.indices()),
            ArraySlot::Boundaries => format!("{:?}", obs.boundaries()),
            // Complexes don't have a nice repr in Rust, so just delegate the whole load to Python
            // and convert back.
            ArraySlot::Coeffs => PyList::new_bound(py, obs.coeffs()).repr()?.to_string(),
            ArraySlot::BitTerms => format!(
                "[{}]",
                obs.bit_terms()
                    .iter()
                    .map(BitTerm::py_label)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<observable {} view: {}>",
            match self.slot {
                ArraySlot::Coeffs => "coeffs",
                ArraySlot::BitTerms => "bit_terms",
                ArraySlot::Indices => "indices",
                ArraySlot::Boundaries => "boundaries",
            },
            data,
        ))
    }

    fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<Py<PyAny>> {
        // The slightly verbose generic setup here is to allow the type of a scalar return to be
        // different to the type that gets put into the Numpy array, since the `BitTerm` enum can be
        // a direct scalar, but for Numpy, we need it to be a raw `u8`.
        fn get_from_slice<T, S>(
            py: Python,
            slice: &[T],
            index: PySequenceIndex,
        ) -> PyResult<Py<PyAny>>
        where
            T: ToPyObject + Copy + Into<S>,
            S: ::numpy::Element,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => Ok(slice[index].to_object(py)),
                indices => Ok(PyArray1::from_iter_bound(
                    py,
                    indices.iter().map(|index| slice[index].into()),
                )
                .into_any()
                .unbind()),
            }
        }

        let borrowed = self.base.borrow(py);
        let obs = borrowed.inner.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => get_from_slice::<_, Complex64>(py, obs.coeffs(), index),
            ArraySlot::BitTerms => get_from_slice::<_, u8>(py, obs.bit_terms(), index),
            ArraySlot::Indices => get_from_slice::<_, u32>(py, obs.indices(), index),
            ArraySlot::Boundaries => get_from_slice::<_, usize>(py, obs.boundaries(), index),
        }
    }

    fn __setitem__(&self, index: PySequenceIndex, values: &Bound<PyAny>) -> PyResult<()> {
        /// Set values of a slice according to the indexer, using `extract` to retrieve the
        /// Rust-space object from the collection of Python-space values.
        ///
        /// This indirects the Python extraction through an intermediate type to marginally improve
        /// the error messages for things like `BitTerm`, where Python-space extraction might fail
        /// because the user supplied an invalid alphabet letter.
        ///
        /// This allows broadcasting a single item into many locations in a slice (like Numpy), but
        /// otherwise requires that the index and values are the same length (unlike Python's
        /// `list`) because that would change the length.
        fn set_in_slice<'py, T, S>(
            slice: &mut [T],
            index: PySequenceIndex<'py>,
            values: &Bound<'py, PyAny>,
        ) -> PyResult<()>
        where
            T: Copy + TryFrom<S>,
            S: FromPyObject<'py>,
            PyErr: From<<T as TryFrom<S>>::Error>,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => {
                    slice[index] = values.extract::<S>()?.try_into()?;
                    Ok(())
                }
                indices => {
                    if let Ok(value) = values.extract::<S>() {
                        let value = value.try_into()?;
                        for index in indices {
                            slice[index] = value;
                        }
                    } else {
                        let values = values
                            .iter()?
                            .map(|value| value?.extract::<S>()?.try_into().map_err(PyErr::from))
                            .collect::<PyResult<Vec<_>>>()?;
                        if indices.len() != values.len() {
                            return Err(PyValueError::new_err(format!(
                                "tried to set a slice of length {} with a sequence of length {}",
                                indices.len(),
                                values.len(),
                            )));
                        }
                        for (index, value) in indices.into_iter().zip(values) {
                            slice[index] = value;
                        }
                    }
                    Ok(())
                }
            }
        }

        let borrowed = self.base.borrow(values.py());
        let mut obs = borrowed.inner.write().map_err(|_| InnerWriteError)?;
        match self.slot {
            ArraySlot::Coeffs => set_in_slice::<_, Complex64>(obs.coeffs_mut(), index, values),
            ArraySlot::BitTerms => set_in_slice::<BitTerm, u8>(obs.bit_terms_mut(), index, values),
            ArraySlot::Indices => set_in_slice::<_, u32>(obs.indices_mut(), index, values),
            ArraySlot::Boundaries => set_in_slice::<_, usize>(obs.boundaries_mut(), index, values),
        }
    }

    fn __len__(&self, py: Python) -> PyResult<usize> {
        let borrowed = self.base.borrow(py);
        let obs = borrowed.inner.read().map_err(|_| InnerReadError)?;
        let len = match self.slot {
            ArraySlot::Coeffs => obs.coeffs().len(),
            ArraySlot::BitTerms => obs.bit_terms().len(),
            ArraySlot::Indices => obs.indices().len(),
            ArraySlot::Boundaries => obs.boundaries().len(),
        };
        Ok(len)
    }

    #[pyo3(signature = (/, dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // This method always copies, so we don't leave dangling pointers lying around in Numpy
        // arrays; it's not enough just to set the `base` of the Numpy array to the
        // `SparseObservable`, since the `Vec` we're referring to might re-allocate and invalidate
        // the pointer the Numpy array is wrapping.
        if !copy.unwrap_or(true) {
            return Err(PyValueError::new_err(
                "cannot produce a safe view onto movable memory",
            ));
        }
        let borrowed = self.base.borrow(py);
        let obs = borrowed.inner.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => {
                cast_array_type(py, PyArray1::from_slice_bound(py, obs.coeffs()), dtype)
            }
            ArraySlot::Indices => {
                cast_array_type(py, PyArray1::from_slice_bound(py, obs.indices()), dtype)
            }
            ArraySlot::Boundaries => {
                cast_array_type(py, PyArray1::from_slice_bound(py, obs.boundaries()), dtype)
            }
            ArraySlot::BitTerms => {
                let bit_terms: &[u8] = ::bytemuck::cast_slice(obs.bit_terms());
                cast_array_type(py, PyArray1::from_slice_bound(py, bit_terms), dtype)
            }
        }
    }
}

/// Use the Numpy Python API to convert a `PyArray` into a dynamically chosen `dtype`, copying only
/// if required.
fn cast_array_type<'py, T>(
    py: Python<'py>,
    array: Bound<'py, PyArray1<T>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let base_dtype = array.dtype();
    let dtype = dtype
        .map(|dtype| PyArrayDescr::new_bound(py, dtype))
        .unwrap_or_else(|| Ok(base_dtype.clone()))?;
    if dtype.is_equiv_to(&base_dtype) {
        return Ok(array.into_any());
    }
    PyModule::import_bound(py, intern!(py, "numpy"))?
        .getattr(intern!(py, "array"))?
        .call(
            (array,),
            Some(
                &[
                    (intern!(py, "copy"), NUMPY_COPY_ONLY_IF_NEEDED.get_bound(py)),
                    (intern!(py, "dtype"), dtype.as_any()),
                ]
                .into_py_dict_bound(py),
            ),
        )
        .map(|obj| obj.into_any())
}

/// Attempt to coerce an arbitrary Python object to a [PySparseObservable].
///
/// This returns:
///
/// * `Ok(Some(obs))` if the coercion was completely successful.
/// * `Ok(None)` if the input value was just completely the wrong type and no coercion could be
///    attempted.
/// * `Err` if the input was a valid type for coercion, but the coercion failed with a Python
///   exception.
///
/// The purpose of this is for conversion the arithmetic operations, which should return
/// [PyNotImplemented] if the type is not valid for coercion.
fn coerce_to_observable<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PySparseObservable>>> {
    let py = value.py();
    if let Ok(obs) = value.downcast_exact::<PySparseObservable>() {
        return Ok(Some(obs.clone()));
    }
    match PySparseObservable::py_new(value, None) {
        Ok(obs) => Ok(Some(Bound::new(py, obs)?)),
        Err(e) => {
            if e.is_instance_of::<PyTypeError>(py) {
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}
pub fn py_sparse_observable(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PySparseObservable>()?;
    Ok(())
}
