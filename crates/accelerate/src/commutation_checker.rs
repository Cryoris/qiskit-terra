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

use approx::abs_diff_eq;
use hashbrown::hash_map::Iter;
use hashbrown::{HashMap, HashSet};
use ndarray::linalg::kron;
use ndarray::Array2;
use num_bigint::BigInt;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::intern;
use smallvec::SmallVec;

use crate::unitary_compose::compose;
use crate::QiskitError;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PySequence, PyTuple};
use qiskit_circuit::bit_data::BitData;
use qiskit_circuit::circuit_instruction::{ExtraInstructionAttributes, OperationFromPython};
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::operations::OperationRef::{Gate as PyGateType, Operation as PyOperationType};
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_circuit::{BitType, Clbit, Qubit};

static SKIPPED_NAMES: [&str; 4] = ["measure", "reset", "delay", "initialize"];
static NO_CACHE_NAMES: [&str; 2] = ["annotated", "linear_function"];
static SUPPORTED_OP: Lazy<HashSet<&str>> = Lazy::new(|| {
    HashSet::from([
        "h", "x", "y", "z", "sx", "sxdg", "t", "tdg", "s", "sdg", "cx", "cy", "cz", "swap",
        "iswap", "ecr", "ccx", "cswap",
    ])
});

#[pyclass(module = "qiskit._accelerate.commutation_checker")]
struct CommutationChecker {
    library: CommutationLibrary,
    cache_max_entries: usize,
    cache: HashMap<(String, String), CommutationCacheEntry>,
    #[pyo3(get)]
    current_cache_entries: usize,
    #[pyo3(get)]
    _cache_miss: usize,
    #[pyo3(get)]
    _cache_hit: usize,
    #[pyo3(get)]
    gates: Option<HashSet<String>>,
}

#[pymethods]
impl CommutationChecker {
    #[pyo3(signature = (standard_gate_commutations=None, cache_max_entries=1_000_000, gates=None))]
    #[new]
    fn py_new(
        standard_gate_commutations: Option<Bound<PyAny>>, // Send a bound here
        cache_max_entries: usize,
        gates: Option<HashSet<String>>,
    ) -> Self {
        // Initialize sets before they are used in the commutation checker
        Lazy::force(&SUPPORTED_OP);
        CommutationChecker {
            library: CommutationLibrary::new(standard_gate_commutations),
            cache: HashMap::with_capacity(cache_max_entries),
            cache_max_entries,
            current_cache_entries: 0,
            _cache_miss: 0,
            _cache_hit: 0,
            gates,
        }
    }

    #[pyo3(signature=(op1, op2, max_num_qubits=3))]
    fn commute_nodes(
        &mut self,
        py: Python,
        op1: &DAGOpNode,
        op2: &DAGOpNode,
        max_num_qubits: u32,
    ) -> PyResult<bool> {
        let (qargs1, qargs2) = get_bits::<Qubit>(
            py,
            op1.instruction.qubits.bind(py),
            op2.instruction.qubits.bind(py),
        )?;
        let (cargs1, cargs2) = get_bits::<Clbit>(
            py,
            op1.instruction.clbits.bind(py),
            op2.instruction.clbits.bind(py),
        )?;

        self.commute_inner(
            py,
            &op1.instruction.operation.view(),
            &op1.instruction.params,
            op1.instruction.extra_attrs.as_deref(),
            &qargs1,
            &cargs1,
            &op2.instruction.operation.view(),
            &op2.instruction.params,
            op2.instruction.extra_attrs.as_deref(),
            &qargs2,
            &cargs2,
            max_num_qubits,
        )
    }

    #[pyo3(signature=(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits=3))]
    #[allow(clippy::too_many_arguments)]
    fn commute(
        &mut self,
        py: Python,
        op1: OperationFromPython,
        qargs1: Option<&Bound<PySequence>>,
        cargs1: Option<&Bound<PySequence>>,
        op2: OperationFromPython,
        qargs2: Option<&Bound<PySequence>>,
        cargs2: Option<&Bound<PySequence>>,
        max_num_qubits: u32,
    ) -> PyResult<bool> {
        let qargs1 =
            qargs1.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs1 =
            cargs1.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let qargs2 =
            qargs2.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs2 =
            cargs2.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;

        let (qargs1, qargs2) = get_bits::<Qubit>(py, &qargs1, &qargs2)?;
        let (cargs1, cargs2) = get_bits::<Clbit>(py, &cargs1, &cargs2)?;

        self.commute_inner(
            py,
            &op1.operation.view(),
            &op1.params,
            op1.extra_attrs.as_deref(),
            &qargs1,
            &cargs1,
            &op2.operation.view(),
            &op2.params,
            op2.extra_attrs.as_deref(),
            &qargs2,
            &cargs2,
            max_num_qubits,
        )
    }

    #[pyo3(signature=())]
    fn num_cached_entries(&self) -> usize {
        self.current_cache_entries
    }
    #[pyo3(signature=())]
    fn clear_cached_commutations(&mut self) {
        self.clear_cache()
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new_bound(py);
        out_dict.set_item("cache_max_entries", self.cache_max_entries)?;
        out_dict.set_item("current_cache_entries", self.current_cache_entries)?;
        out_dict.set_item("_cache_miss", self._cache_miss)?;
        out_dict.set_item("_cache_hit", self._cache_hit)?;
        out_dict.set_item("cache", self.cache.clone())?;
        out_dict.set_item("library", self.library.library.to_object(py))?;
        out_dict.set_item("gates", self.gates.clone())?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.downcast_bound::<PyDict>(py)?;
        self.cache_max_entries = dict_state
            .get_item("cache_max_entries")?
            .unwrap()
            .extract()?;
        self.current_cache_entries = dict_state
            .get_item("current_cache_entries")?
            .unwrap()
            .extract()?;
        self._cache_miss = dict_state.get_item("_cache_miss")?.unwrap().extract()?;
        self._cache_hit = dict_state.get_item("_cache_hit")?.unwrap().extract()?;
        self.library = CommutationLibrary {
            library: dict_state.get_item("library")?.unwrap().extract()?,
        };
        self.cache = dict_state.get_item("cache")?.unwrap().extract()?;
        self.gates = dict_state.get_item("gates")?.unwrap().extract()?;
        Ok(())
    }
}

impl CommutationChecker {
    #[allow(clippy::too_many_arguments)]
    fn commute_inner(
        &mut self,
        py: Python,
        op1: &OperationRef,
        params1: &[Param],
        attrs1: Option<&ExtraInstructionAttributes>,
        qargs1: &[Qubit],
        cargs1: &[Clbit],
        op2: &OperationRef,
        params2: &[Param],
        attrs2: Option<&ExtraInstructionAttributes>,
        qargs2: &[Qubit],
        cargs2: &[Clbit],
        max_num_qubits: u32,
    ) -> PyResult<bool> {
        if let Some(gates) = &self.gates {
            if !gates.is_empty() && (!gates.contains(op1.name()) || !gates.contains(op2.name())) {
                return Ok(false);
            }
        }

        let commutation: Option<bool> = commutation_precheck(
            op1,
            params1,
            attrs1,
            qargs1,
            cargs1,
            op2,
            params2,
            attrs2,
            qargs2,
            cargs2,
            max_num_qubits,
        );
        if let Some(is_commuting) = commutation {
            return Ok(is_commuting);
        }

        let reversed = if op1.num_qubits() != op2.num_qubits() {
            op1.num_qubits() > op2.num_qubits()
        } else {
            BigInt::from_signed_bytes_be(op1.name().as_bytes())
                >= BigInt::from_signed_bytes_be(op2.name().as_bytes())
        };
        let (first_params, second_params) = if reversed {
            (params2, params1)
        } else {
            (params1, params2)
        };
        let (first_op, second_op) = if reversed { (op2, op1) } else { (op1, op2) };
        let (first_qargs, second_qargs) = if reversed {
            (qargs2, qargs1)
        } else {
            (qargs1, qargs2)
        };

        let skip_cache: bool = NO_CACHE_NAMES.contains(&first_op.name()) ||
            NO_CACHE_NAMES.contains(&second_op.name()) ||
            // Skip params that do not evaluate to floats for caching and commutation library
            first_params.iter().any(|p| !matches!(p, Param::Float(_))) ||
            second_params.iter().any(|p| !matches!(p, Param::Float(_)));

        if skip_cache {
            return self.commute_matmul(
                py,
                first_op,
                first_params,
                first_qargs,
                second_op,
                second_params,
                second_qargs,
            );
        }

        // Query commutation library
        if let Some(is_commuting) =
            self.library
                .check_commutation_entries(first_op, first_qargs, second_op, second_qargs)
        {
            return Ok(is_commuting);
        }
        // Query cache
        match self
            .cache
            .get(&(first_op.name().to_string(), second_op.name().to_string()))
        {
            Some(commutation_dict) => {
                let placement = get_relative_placement(first_qargs, second_qargs);
                let hashes = (
                    hashable_params(&first_params),
                    hashable_params(&second_params),
                );
                match commutation_dict.get(&(placement, hashes)) {
                    Some(commutation) => {
                        self._cache_hit += 1;
                        return Ok(*commutation);
                    }
                    None => self._cache_miss += 1,
                }
            }
            None => self._cache_miss += 1,
        }

        // Perform matrix multiplication to determine commutation
        let is_commuting = self.commute_matmul(
            py,
            first_op,
            first_params,
            first_qargs,
            second_op,
            second_params,
            second_qargs,
        )?;

        // TODO: implement a LRU cache for this
        if self.current_cache_entries >= self.cache_max_entries {
            self.clear_cache();
        }
        // Cache results from is_commuting
        self.cache
            .entry((first_op.name().to_string(), second_op.name().to_string()))
            .and_modify(|entries| {
                let key = (
                    get_relative_placement(first_qargs, second_qargs),
                    (
                        hashable_params(&first_params),
                        hashable_params(&second_params),
                    ),
                );
                entries.insert(key, is_commuting);
                self.current_cache_entries += 1;
            })
            .or_insert_with(|| {
                let mut entries = HashMap::with_capacity(1);
                let key = (
                    get_relative_placement(first_qargs, second_qargs),
                    (
                        hashable_params(&first_params),
                        hashable_params(&second_params),
                    ),
                );
                entries.insert(key, is_commuting);
                self.current_cache_entries += 1;
                CommutationCacheEntry { mapping: entries }
            });
        Ok(is_commuting)
    }

    fn commute_matmul(
        &self,
        py: Python,
        first_op: &OperationRef,
        first_params: &[Param],
        first_qargs: &[Qubit],
        second_op: &OperationRef,
        second_params: &[Param],
        second_qargs: &[Qubit],
    ) -> PyResult<bool> {
        // compute relative positioning of qargs of the second gate to the first gate
        let mut qarg: HashMap<&Qubit, Qubit> =
            HashMap::with_capacity(first_qargs.len() + second_qargs.len());
        for (i, q) in first_qargs.iter().enumerate() {
            qarg.entry(q).or_insert(Qubit(i as u32));
        }
        let mut num_qubits = first_qargs.len() as u32;
        for q in second_qargs {
            if !qarg.contains_key(q) {
                qarg.insert(q, Qubit(num_qubits));
                num_qubits += 1;
            }
        }

        let first_qarg: Vec<_> = first_qargs.iter().map(|q| *qarg.get(q).unwrap()).collect();
        let second_qarg: Vec<_> = second_qargs.iter().map(|q| *qarg.get(q).unwrap()).collect();

        if first_qarg.len() > second_qarg.len() {
            return Err(QiskitError::new_err(
                "first instructions must have at most as many qubits as the second instruction",
            ));
        };

        let first_mat = match get_matrix(py, first_op, first_params) {
            Some(matrix) => matrix,
            None => return Ok(false),
        };

        let second_mat = match get_matrix(py, second_op, second_params) {
            Some(matrix) => matrix,
            None => return Ok(false),
        };

        if first_qarg == second_qarg {
            Ok(abs_diff_eq!(
                second_mat.dot(&first_mat),
                first_mat.dot(&second_mat),
                epsilon = 1e-8
            ))
        } else {
            let extra_qarg2 = num_qubits - first_qarg.len() as u32;
            let first_mat = if extra_qarg2 > 0 {
                let id_op = Array2::<Complex64>::eye(usize::pow(2, extra_qarg2));
                kron(&id_op, &first_mat)
            } else {
                first_mat
            };
            let op12 = compose(&first_mat.view(), &second_mat.view(), &second_qarg, false);
            let op21 = compose(&first_mat.view(), &second_mat.view(), &second_qarg, true);
            Ok(abs_diff_eq!(op12, op21, epsilon = 1e-8))
        }
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
        self.current_cache_entries = 0;
        self._cache_miss = 0;
        self._cache_hit = 0;
    }
}

fn commutation_precheck(
    op1: &OperationRef,
    params1: &[Param],
    attrs1: Option<&ExtraInstructionAttributes>,
    qargs1: &[Qubit],
    cargs1: &[Clbit],
    op2: &OperationRef,
    params2: &[Param],
    attrs2: Option<&ExtraInstructionAttributes>,
    qargs2: &[Qubit],
    cargs2: &[Clbit],
    max_num_qubits: u32,
) -> Option<bool> {
    if op1.control_flow()
        || op2.control_flow()
        || attrs1.is_some_and(|attr| attr.condition.is_some())
        || attrs2.is_some_and(|attr| attr.condition.is_some())
    {
        return Some(false);
    }

    // assuming the number of involved qubits to be small, this might be faster than set operations
    if !qargs1.iter().any(|e| qargs2.contains(e)) && !cargs1.iter().any(|e| cargs2.contains(e)) {
        return Some(true);
    }

    if qargs1.len() > max_num_qubits as usize || qargs2.len() > max_num_qubits as usize {
        return Some(false);
    }

    if SUPPORTED_OP.contains(op1.name()) && SUPPORTED_OP.contains(op2.name()) {
        return None;
    }

    if is_commutation_skipped(op1, params1) || is_commutation_skipped(op2, params2) {
        return Some(false);
    }

    None
}

fn get_matrix(py: Python, operation: &OperationRef, params: &[Param]) -> Option<Array2<Complex64>> {
    match operation.matrix(params) {
        Some(matrix) => Some(matrix),
        None => match operation {
            PyGateType(gate) => matrix_via_operator(py, &gate.gate),
            PyOperationType(op) => matrix_via_operator(py, &op.operation),
            _ => None,
        },
    }
}
fn matrix_via_operator(py: Python, py_obj: &PyObject) -> Option<Array2<Complex64>> {
    Some(
        QI_OPERATOR
            .get_bound(py)
            .call1((py_obj,))
            .ok()?
            .getattr(intern!(py, "data"))
            .ok()?
            .extract::<PyReadonlyArray2<Complex64>>()
            .ok()?
            .as_array()
            .to_owned(),
    )
}

fn is_commutation_skipped<T>(op: &T, params: &[Param]) -> bool
where
    T: Operation,
{
    op.directive()
        || SKIPPED_NAMES.contains(&op.name())
        || params
            .iter()
            .any(|x| matches!(x, Param::ParameterExpression(_)))
}

fn get_relative_placement(
    first_qargs: &[Qubit],
    second_qargs: &[Qubit],
) -> SmallVec<[Option<Qubit>; 2]> {
    let qubits_g2: HashMap<_, _> = second_qargs
        .iter()
        .enumerate()
        .map(|(i_g1, q_g1)| (q_g1, Qubit(i_g1 as u32)))
        .collect();

    first_qargs
        .iter()
        .map(|q_g0| qubits_g2.get(q_g0).copied())
        .collect()
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct CommutationLibrary {
    pub library: Option<HashMap<(String, String), CommutationLibraryEntry>>,
}

impl CommutationLibrary {
    fn check_commutation_entries(
        &self,
        first_op: &OperationRef,
        first_qargs: &[Qubit],
        second_op: &OperationRef,
        second_qargs: &[Qubit],
    ) -> Option<bool> {
        if let Some(library) = &self.library {
            match library.get(&(first_op.name().to_string(), second_op.name().to_string())) {
                Some(CommutationLibraryEntry::Commutes(b)) => Some(*b),
                Some(CommutationLibraryEntry::QubitMapping(qm)) => qm
                    .get(&get_relative_placement(first_qargs, second_qargs))
                    .copied(),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[pymethods]
impl CommutationLibrary {
    #[new]
    fn new(py_any: Option<Bound<PyAny>>) -> Self {
        match py_any {
            Some(pyob) => CommutationLibrary {
                library: pyob
                    .extract::<Option<HashMap<(String, String), CommutationLibraryEntry>>>()
                    .unwrap(),
            },
            None => CommutationLibrary {
                library: Some(HashMap::new()),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum CommutationLibraryEntry {
    Commutes(bool),
    QubitMapping(HashMap<SmallVec<[Option<Qubit>; 2]>, bool>),
}

impl<'py> FromPyObject<'py> for CommutationLibraryEntry {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        if let Ok(b) = b.extract::<bool>() {
            return Ok(CommutationLibraryEntry::Commutes(b));
        }
        let dict = b.downcast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in dict {
            let raw_key: SmallVec<[Option<u32>; 2]> = k.extract()?;
            let v: bool = v.extract()?;
            let key = raw_key.into_iter().map(|key| key.map(Qubit)).collect();
            ret.insert(key, v);
        }
        Ok(CommutationLibraryEntry::QubitMapping(ret))
    }
}

impl ToPyObject for CommutationLibraryEntry {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            CommutationLibraryEntry::Commutes(b) => b.into_py(py),
            CommutationLibraryEntry::QubitMapping(qm) => {
                let out_dict = PyDict::new_bound(py);

                qm.iter().for_each(|(k, v)| {
                    out_dict
                        .set_item(
                            PyTuple::new_bound(py, k.iter().map(|q| q.map(|t| t.0))),
                            PyBool::new_bound(py, *v),
                        )
                        .ok()
                        .unwrap()
                });
                out_dict.unbind().into_any()
            }
        }
    }
}

type CacheKey = (
    SmallVec<[Option<Qubit>; 2]>,
    (SmallVec<[ParameterKey; 3]>, SmallVec<[ParameterKey; 3]>),
);
// Need a struct instead of a type definition because we cannot implement serialization traits otherwise
#[derive(Clone, Debug)]
struct CommutationCacheEntry {
    mapping: HashMap<CacheKey, bool>,
}
impl CommutationCacheEntry {
    fn get(&self, key: &CacheKey) -> Option<&bool> {
        self.mapping.get(key)
    }
    fn iter(&self) -> Iter<'_, CacheKey, bool> {
        self.mapping.iter()
    }

    fn insert(&mut self, k: CacheKey, v: bool) -> Option<bool> {
        self.mapping.insert(k, v)
    }
}

impl ToPyObject for CommutationCacheEntry {
    fn to_object(&self, py: Python) -> PyObject {
        let out_dict = PyDict::new_bound(py);
        for (k, v) in self.iter() {
            let qubits = PyTuple::new_bound(py, k.0.iter().map(|q| q.map(|t| t.0)));
            let params0 = PyTuple::new_bound(py, k.1 .0.iter().map(|pk| pk.0));
            let params1 = PyTuple::new_bound(py, k.1 .1.iter().map(|pk| pk.0));
            out_dict
                .set_item(
                    PyTuple::new_bound(py, [qubits, PyTuple::new_bound(py, [params0, params1])]),
                    PyBool::new_bound(py, *v),
                )
                .expect("Failed to construct commutation cache for serialization");
        }
        out_dict.into_any().unbind()
    }
}

type CacheKeyRaw = (
    SmallVec<[Option<u32>; 2]>,
    (SmallVec<[f64; 3]>, SmallVec<[f64; 3]>),
);
impl<'py> FromPyObject<'py> for CommutationCacheEntry {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        let dict = b.downcast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in dict {
            let raw_key: CacheKeyRaw = k.extract()?;
            let qubits = raw_key.0.iter().map(|q| q.map(Qubit)).collect();
            let params0: SmallVec<_> = raw_key.1 .0.iter().map(|p| ParameterKey(*p)).collect();
            let params1: SmallVec<_> = raw_key.1 .1.iter().map(|p| ParameterKey(*p)).collect();
            let v: bool = v.extract()?;
            ret.insert((qubits, (params0, params1)), v);
        }
        Ok(CommutationCacheEntry { mapping: ret })
    }
}

#[derive(Debug, Copy, Clone)]
struct ParameterKey(f64);

impl ParameterKey {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
}

impl std::hash::Hash for ParameterKey {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for ParameterKey {
    fn eq(&self, other: &ParameterKey) -> bool {
        self.key() == other.key()
    }
}

impl Eq for ParameterKey {}

fn hashable_params(params: &[Param]) -> SmallVec<[ParameterKey; 3]> {
    params
        .iter()
        .map(|x| {
            if let Param::Float(x) = x {
                ParameterKey(*x)
            } else {
                panic!("Unable to hash a non-float instruction parameter.")
            }
        })
        .collect()
}

#[pymodule]
pub fn commutation_checker(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CommutationLibrary>()?;
    m.add_class::<CommutationChecker>()?;
    Ok(())
}

fn get_bits<T>(
    py: Python,
    bits1: &Bound<PyTuple>,
    bits2: &Bound<PyTuple>,
) -> PyResult<(Vec<T>, Vec<T>)>
where
    T: From<BitType> + Copy,
    BitType: From<T>,
{
    let mut bitdata: BitData<T> = BitData::new(py, "bits".to_string());

    bits1.iter().chain(bits2.iter()).for_each(|bit| {
        bitdata.add(py, &bit, false).unwrap();
    });

    Ok((
        bitdata.map_bits(bits1)?.collect(),
        bitdata.map_bits(bits2)?.collect(),
    ))
}
