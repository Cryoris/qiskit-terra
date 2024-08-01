# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""

High Level Synthesis Plugins
-----------------------------

Clifford Synthesis
''''''''''''''''''

.. list-table:: Plugins for :class:`qiskit.quantum_info.Clifford` (key = ``"clifford"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"ag"``
      - :class:`~.AGSynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count
    * - ``"bm"``
      - :class:`~.BMSynthesisClifford`
      - all-to-all
      - optimal count for `n=2,3`; used in ``"default"`` for `n=2,3`
    * - ``"greedy"``
      - :class:`~.GreedySynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"`` for `n>=4`
    * - ``"layers"``
      - :class:`~.LayerSynthesisClifford`
      - all-to-all
      -
    * - ``"lnn"``
      - :class:`~.LayerLnnSynthesisClifford`
      - linear
      - many CX-gates but guarantees CX-depth of at most `7*n+2`
    * - ``"default"``
      - :class:`~.DefaultSynthesisClifford`
      - all-to-all
      - usually best for optimizing CX-count (and optimal CX-count for `n=2,3`)

.. autosummary::
   :toctree: ../stubs/

   AGSynthesisClifford
   BMSynthesisClifford
   GreedySynthesisClifford
   LayerSynthesisClifford
   LayerLnnSynthesisClifford
   DefaultSynthesisClifford


Linear Function Synthesis
'''''''''''''''''''''''''

.. list-table:: Plugins for :class:`.LinearFunction` (key = ``"linear"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"kms"``
      - :class:`~.KMSSynthesisLinearFunction`
      - linear
      - many CX-gates but guarantees CX-depth of at most `5*n`
    * - ``"pmh"``
      - :class:`~.PMHSynthesisLinearFunction`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"``
    * - ``"default"``
      - :class:`~.DefaultSynthesisLinearFunction`
      - all-to-all
      - best for optimizing CX-count

.. autosummary::
   :toctree: ../stubs/

   KMSSynthesisLinearFunction
   PMHSynthesisLinearFunction
   DefaultSynthesisLinearFunction


Permutation Synthesis
'''''''''''''''''''''

.. list-table:: Plugins for :class:`.PermutationGate` (key = ``"permutation"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"basic"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - optimal SWAP-count; used in ``"default"``
    * - ``"acg"``
      - :class:`~.ACGSynthesisPermutation`
      - all-to-all
      - guarantees SWAP-depth of at most `2`
    * - ``"kms"``
      - :class:`~.KMSSynthesisPermutation`
      - linear
      - many SWAP-gates, but guarantees SWAP-depth of at most `n`
    * - ``"token_swapper"``
      - :class:`~.TokenSwapperSynthesisPermutation`
      - any
      - greedily optimizes SWAP-count for arbitrary connectivity
    * - ``"default"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - best for optimizing SWAP-count

.. autosummary::
   :toctree: ../stubs/

   BasicSynthesisPermutation
   ACGSynthesisPermutation
   KMSSynthesisPermutation
   TokenSwapperSynthesisPermutation


QFT Synthesis
'''''''''''''

.. list-table:: Plugins for :class:`.QFTGate` (key = ``"qft"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
    * - ``"full"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all
    * - ``"line"``
      - :class:`~.QFTSynthesisLine`
      - linear
    * - ``"default"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all

.. autosummary::
   :toctree: ../stubs/

   QFTSynthesisFull
   QFTSynthesisLine
"""

from __future__ import annotations

import typing
from typing import Optional, Union, List, Callable, Sequence
from dataclasses import dataclass

import numpy as np
import rustworkx as rx

from qiskit.circuit.annotated_operation import Modifier
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import (
    ControlFlowOp,
    ControlledGate,
    EquivalenceLibrary,
    Barrier,
    Delay,
    Reset,
    equivalence,
)
from qiskit.circuit.library import LinearFunction, IGate
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper

from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.library import QFTGate
from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_layers,
    synth_clifford_depth_lnn,
    synth_clifford_greedy,
    synth_clifford_ag,
    synth_clifford_bm,
)
from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
    calc_inverse_matrix,
)
from qiskit.synthesis.linear.linear_circuits_utils import transpose_cx_circ
from qiskit.synthesis.permutation import (
    synth_permutation_basic,
    synth_permutation_acg,
    synth_permutation_depth_lnn_kms,
)
from qiskit.synthesis.qft import (
    synth_qft_full,
    synth_qft_line,
)

from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(
        self,
        use_default_on_unspecified: bool = True,
        plugin_selection: str = "sequential",
        plugin_evaluation_fn: Optional[Callable[[QuantumCircuit], int]] = None,
        **kwargs,
    ):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified: if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            plugin_selection: if set to ``"sequential"`` (default), for every higher-level-object
                the synthesis pass will consider the specified methods sequentially, stopping
                at the first method that is able to synthesize the object. If set to ``"all"``,
                all the specified methods will be considered, and the best synthesized circuit,
                according to ``plugin_evaluation_fn`` will be chosen.
            plugin_evaluation_fn: a callable that evaluates the quality of the synthesized
                quantum circuit; a smaller value means a better circuit. If ``None``, the
                quality of the circuit its size (i.e. the number of gates that it contains).
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.plugin_selection = plugin_selection
        self.plugin_evaluation_fn = (
            plugin_evaluation_fn if plugin_evaluation_fn is not None else lambda qc: qc.size()
        )
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


def instruction_to_circuit(inst: Instruction) -> QuantumCircuit:
    circuit = QuantumCircuit(inst.num_qubits, inst.num_clbits)
    circuit.append(inst, circuit.qubits, circuit.clbits)
    return circuit


@dataclass
class QubitTracker:
    """Track qubits per index as and their state."""

    qubits: tuple[int]
    clean: set[int]
    dirty: set[int]

    def num_of_state(self, state: bool, active_qubits: list[int] | None = None):
        """Return the number of qubits with given ``state``.

        Args:
            active_qubits: If given, these are qubits involved in an operation and will
                not be returned.
        """
        if active_qubits is None:
            states = list(self.states.values())
        else:
            states = [state for qubit, state in self.states.items() if qubit not in active_qubits]

        return states.count(state)

    def num_clean(self, active_qubits: set[int] | None = None):
        return self.clean.difference(active_qubits or set())

    def num_dirty(self, active_qubits: set[int] | None = None):
        return self.dirty.difference(active_qubits or set())

    def borrow(self, num_qubits: int, active_qubits: list[int] | None = None) -> list[Qubit]:
        """Get ``num_qubits`` qubits, excluding ``active_qubits``."""
        if active_qubits is None:
            active_qubits = []

        if num_qubits > (available := len(self.qubits) - len(active_qubits)):
            raise RuntimeError(f"Cannot borrow {num_qubits} qubits, only {available} available.")

        return [qubit for qubit in self.qubits if qubit not in active_qubits][:num_qubits]

    def used(self, qubits: set[int]) -> None:
        """Set the state of ``qubits`` to used (i.e. False)."""
        self.clean.discard(qubits)
        self.dirty.update(qubits)
        # could add checks
        # for qubit in qubits:
        #     if qubit not in self.states:
        #         raise KeyError(f"Setting state of unknown qubit {qubit}.")

    def reset(self, qubits: set[int]) -> None:
        """Set the state of ``qubits`` to 0 (i.e. True)."""
        self.dirty.discard(qubits)
        self.clean.update(qubits)
        # for qubit in qubits:
        #     if qubit not in self.states:
        #         raise KeyError(f"Setting state of unknown qubit {qubit}.")
        #     self.states[qubit] = True

    def copy(self, qubit_map: dict[int, int] | None = None) -> "QubitTracker":
        """Copy self.

        Args:
            qubit_map: If provided, apply the mapping ``{old_qubit: new_qubit}`` to
                the qubits in the tracker.
        """
        if qubit_map is None:
            clean = self.clean.copy()
            dirty = self.dirty.copy()
            qubits = self.qubits  # tuple is immutable, no need to copy
        else:
            raise NotImplementedError("currently broken yo")
            qubits = tuple(qubit_map.values())
            states = {
                new_qubit: self.states[old_qubit] for old_qubit, new_qubit in qubit_map.items()
            }

        return QubitTracker(qubits, clean=clean, dirty=dirty)


class HighLevelSynthesis(TransformationPass):
    r"""Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.

    The high-level-synthesis passes information about available auxiliary qubits, and whether their
    state is clean (defined as :math:`|0\rangle`) or dirty (unknown state) to the synthesis routine
    via the respective arguments ``"num_clean_ancillas"`` and ``"num_dirty_ancillas"``.
    If ``qubits_initially_zero`` is ``True`` (default), the qubits are assumed to be in the
    :math:`|0\rangle` state. When appending a synthesized block using auxiliary qubits onto the
    circuit, we first use the clean auxiliary qubits.

    .. note::

        Synthesis methods are assumed to maintain the state of the auxiliary qubits.
        Concretely this means that clean auxiliary qubits must still be in the :math:`|0\rangle`
        state after the synthesized block, while dirty auxiliary qubits are re-used only
        as dirty qubits.

    """

    def __init__(
        self,
        hls_config: Optional[HLSConfig] = None,
        coupling_map: Optional[CouplingMap] = None,
        target: Optional[Target] = None,
        use_qubit_indices: bool = False,
        equivalence_library: Optional[EquivalenceLibrary] = None,
        basis_gates: Optional[List[str]] = None,
        min_qubits: int = 0,
        qubits_initially_zero: bool = True,
    ):
        r"""
        HighLevelSynthesis initializer.

        Args:
            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map: Optional, directed graph represented as a coupling map.
            target: Optional, the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
            equivalence_library: The equivalence library used (instructions in this library will not
                be unrolled by this pass).
            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            min_qubits: The minimum number of qubits for operations in the input
                dag to translate.
            qubits_initially_zero: Indicates whether the qubits are initially in the state
                :math:`|0\rangle`. This allows the high-level-synthesis to use clean auxiliary qubits
                (i.e. in the zero state) to synthesize an operation.
        """
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)

        self.hls_plugin_manager = HighLevelSynthesisPluginManager()
        self._coupling_map = coupling_map
        self._target = target
        self._use_qubit_indices = use_qubit_indices
        self.qubits_initially_zero = qubits_initially_zero
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._min_qubits = min_qubits

        self._top_level_only = self._basis_gates is None and self._target is None

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if not self._top_level_only and (self._target is None or self._target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            self._device_insts = basic_insts | set(self._basis_gates)
        else:
            self._device_insts = set()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """
        qubits = range(dag.num_qubits())
        tracker = QubitTracker(tuple(qubits), clean=set(qubits), dirty=set())
        return self._run(dag, tracker)

    def _run(self, dag: DAGCircuit, tracker: QubitTracker) -> DAGCircuit:
        # analyze operations
        synthesized_nodes = {}

        for node in dag.topological_op_nodes():
            if dag.has_calibration_for(node) or len(node.qargs) < self._min_qubits:
                continue

            if node.is_directive():
                continue

            if node.is_control_flow():
                node.op = control_flow.map_blocks(self.run, node.op)
                continue

            # If the layout has been set, we additionally need to keep track of the qubit indices
            # which tell us if an operation is supported on a specific set of qubits. If it is not,
            # we can save this.
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)

            if self._definitely_skip_node(node, qubits):
                continue

            synthesized, used_qubits = self.synthesize_operation(dag, node.op, qubits, tracker)

            if synthesized is not None:
                synthesized_nodes[node] = (synthesized, used_qubits)

                # mark the operation qubits as used
                tracker.used(qubits)
            else:
                # update the auxiliary qubit trackers
                if isinstance(node.op, (IGate, Delay, Barrier)):
                    pass  # tracker not updated, these are no-ops
                elif isinstance(node.op, Reset):
                    tracker.reset(qubits)
                else:  # any other op used the clean state up
                    tracker.used(qubits)

        # we did not change anything just return the input
        if len(synthesized_nodes) == 0:
            return dag

        # Otherwise we will rebuild with the new operations. Note that we could also
        # check if no operation changed in size and substitute in-place, but rebuilding is
        # generally as fast or faster, unless very few operations are changed.
        out = dag.copy_empty_like()
        index_to_qubit = dict(enumerate(dag.qubits))

        for node in dag.topological_op_nodes():
            if node in synthesized_nodes:
                op, qubits = synthesized_nodes[node]
                qargs = tuple(index_to_qubit[index] for index in qubits)
                if isinstance(op, Operation):
                    out.apply_operation_back(op, qargs, cargs=[])
                    continue

                if isinstance(op, QuantumCircuit):
                    op = circuit_to_dag(op)

                if isinstance(op, DAGCircuit):
                    # handle different types of ops
                    out.compose(op, qargs, inplace=True)
                else:
                    raise RuntimeError(f"Unexpected synthesized type: {type(op)}")
                # out.apply_operation_back(op, qubits, cargs=())
            else:
                out.apply_operation_back(node.op, node.qargs, node.cargs, check=False)

        return out

    def synthesize_operation(
        self,
        dag: DAGCircuit,  # this could be avoided if we knew the global qubit index for unroll ops
        operation: Operation,
        qubits: tuple[int],
        tracker: QubitTracker,
    ) -> QuantumCircuit | Operation | DAGCircuit | None:
        # Try to synthesize the operation. We'll go through the following options:
        #  (1) Annotations: if the operator is annotated, synthesize the base operation
        #       and then apply the modifiers.
        #  (2) High-level objects: try running the battery of high-level synthesis plugins
        #  (3) Unrolling custom definitions: try defining the operation if it is not yet
        #       in the set of supported instructions
        # If any of the above were triggered, we will recurse and go again through these steps
        # until no further change occurred. If there was no change, we just return ``None``.
        synthesized = None

        # try synthesizing via AnnotatedOperation
        if isinstance(operation, AnnotatedOperation):
            # The base operation must be synthesized without using potential control qubits
            # used in the modifiers.
            num_ctrl = sum(
                mod.num_ctrl_qubits
                for mod in operation.modifiers
                if isinstance(mod, ControlModifier)
            )
            baseop_qubits = qubits[num_ctrl:]  # reminder: control qubits are the first ones
            baseop_tracker = tracker.copy()
            baseop_tracker.used(qubits[:num_ctrl])  # no access to control qubits

            # get qubits of base operation
            synthesized_base_op, _ = self.synthesize_operation(
                dag, operation.base_op, baseop_qubits, baseop_tracker
            )
            if synthesized_base_op is None:
                synthesized_base_op = operation.base_op
            elif isinstance(synthesized_base_op, DAGCircuit):
                synthesized_base_op = dag_to_circuit(synthesized_base_op)

            synthesized = self._apply_annotations(synthesized_base_op, operation.modifiers)

        # try synthesizing via HLS
        if synthesized is None:
            synthesized = self._synthesize_op_using_plugins(
                operation, qubits, tracker.num_clean(qubits), tracker.num_dirty(qubits)
            )

        # try unrolling custom definitions
        if synthesized is None and not self._top_level_only:
            synthesized = self._unroll_custom_definition(dag, operation, qubits)

        if synthesized is None:
            # if we didn't synthesize, there was nothing to unroll, so just set the used qubits
            used_qubits = qubits

        else:
            # if it has been synthesized, recurse and finally store the decomposition
            if isinstance(synthesized, Operation):
                re_synthesized, qubits = self.synthesize_operation(
                    dag, synthesized, qubits, tracker
                )
                if re_synthesized is not None:
                    synthesized = re_synthesized
                used_qubits = qubits

                # if synthesized.num_qubits != len(qubits):
                # raise RuntimeError("currently cannot annotate HLS stuff")

            elif isinstance(synthesized, QuantumCircuit):
                aux_qubits = tracker.borrow(synthesized.num_qubits - len(qubits), qubits)
                used_qubits = qubits + tuple(aux_qubits)
                as_dag = circuit_to_dag(synthesized)
                # map used qubits to subcircuit
                # qubit_map = {used_qubit: i for i, used_qubit in enumerate(used_qubits)}
                # qubit_map = {}

                synthesized = self._run(as_dag, tracker.copy())
                if synthesized.num_qubits() != len(used_qubits):
                    raise RuntimeError(
                        f"Mismatching number of qubits, using {synthesized.num_qubits()} "
                        "but have {len(used_qubits)}."
                    )

            else:
                raise RuntimeError(f"Unexpected synthesized type: {type(synthesized)}")

        if synthesized is not None and used_qubits is None:
            raise RuntimeError("Failed to find qubit indices on", synthesized)

        return synthesized, used_qubits

    def _unroll_custom_definition(
        self, dag: DAGCircuit, inst: Instruction, qubits: list[int] | None
    ) -> QuantumCircuit | None:
        # check if the operation is already supported natively
        if not (isinstance(inst, ControlledGate) and inst._open_ctrl):
            # include path for when target exists but target.num_qubits is None (BasicSimulator)
            inst_supported = self._instruction_supported(inst.name, qubits)
            if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(inst)):
                return None  # we support this operation already

        # if not, try to get the definition
        try:
            definition = inst.definition
        except (TypeError, AttributeError) as err:
            raise TranspilerError(f"HighLevelSynthesis was unable to define {inst.name}.") from err

        if definition is None:
            raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {inst}.")

        return definition

    def _methods_to_try(self, name: str):
        """Get a sequence of methods to try for a given op name."""
        if (methods := self.hls_config.methods.get(name)) is not None:
            # the operation's name appears in the user-provided config,
            # we use the list of methods provided by the user
            return methods
        if (
            self.hls_config.use_default_on_unspecified
            and "default" in self.hls_plugin_manager.method_names(name)
        ):
            # the operation's name does not appear in the user-specified config,
            # we use the "default" method when instructed to do so and the "default"
            # method is available
            return ["default"]
        return []

    def _synthesize_op_using_plugins(
        self, op: Operation, qubits: List, num_clean_ancillas: int = 0, num_dirty_ancillas: int = 0
    ) -> Union[QuantumCircuit, None]:
        """
        Attempts to synthesize op using plugin mechanism.

        The arguments ``num_clean_ancillas`` and ``num_dirty_ancillas`` specify
        the number of clean and dirty qubits available to synthesize the given
        operation. A synthesis method does not need to use these additional qubits.

        Returns either the synthesized circuit or None (which may occur
        when no synthesis methods is available or specified, or when there is
        an insufficient number of auxiliary qubits).
        """
        hls_plugin_manager = self.hls_plugin_manager

        best_decomposition = None
        best_score = np.inf

        for method in self._methods_to_try(op.name):
            # There are two ways to specify a synthesis method. The more explicit
            # way is to specify it as a tuple consisting of a synthesis algorithm and a
            # list of additional arguments, e.g.,
            #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
            #   ("pmh", {}).
            # When the list of additional arguments is empty, one can also specify
            # just the synthesis algorithm, e.g.,
            #   "pmh".
            if isinstance(method, tuple):
                plugin_specifier, plugin_args = method
            else:
                plugin_specifier = method
                plugin_args = {}

            # There are two ways to specify a synthesis algorithm being run,
            # either by name, e.g. "kms" (which then should be specified in entry_points),
            # or directly as a class inherited from HighLevelSynthesisPlugin (which then
            # does not need to be specified in entry_points).
            if isinstance(plugin_specifier, str):
                if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                    raise TranspilerError(
                        f"Specified method: {plugin_specifier} not found in available "
                        f"plugins for {op.name}"
                    )
                plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
            else:
                plugin_method = plugin_specifier

            # Set the number of available clean and dirty auxiliary qubits via plugin args.
            plugin_args["num_clean_ancillas"] = num_clean_ancillas
            plugin_args["num_dirty_ancillas"] = num_dirty_ancillas

            decomposition = plugin_method.run(
                op,
                coupling_map=self._coupling_map,
                target=self._target,
                qubits=qubits,
                **plugin_args,
            )

            # The synthesis methods that are not suited for the given higher-level-object
            # will return None.
            if decomposition is not None:
                if self.hls_config.plugin_selection == "sequential":
                    # In the "sequential" mode the first successful decomposition is
                    # returned.
                    best_decomposition = decomposition
                    break

                # In the "run everything" mode we update the best decomposition
                # discovered
                current_score = self.hls_config.plugin_evaluation_fn(decomposition)
                if current_score < best_score:
                    best_decomposition = decomposition
                    best_score = current_score

        return best_decomposition

    def _apply_annotations(
        self, synthesized: Operation, modifiers: list[Modifier]
    ) -> QuantumCircuit:
        """
        Recursively synthesizes annotated operations.
        Returns either the synthesized operation or None (which occurs when the operation
        is not an annotated operation).
        """
        for modifier in modifiers:
            if isinstance(modifier, InverseModifier):
                # Both QuantumCircuit and Gate have inverse method
                synthesized = synthesized.inverse()

            elif isinstance(modifier, ControlModifier):
                # Both QuantumCircuit and Gate have control method, however for circuits
                # it is more efficient to avoid constructing the controlled quantum circuit.
                if isinstance(synthesized, QuantumCircuit):
                    synthesized = synthesized.to_gate()

                synthesized = synthesized.control(
                    num_ctrl_qubits=modifier.num_ctrl_qubits,
                    label=None,
                    ctrl_state=modifier.ctrl_state,
                    annotated=False,
                )

                if isinstance(synthesized, AnnotatedOperation):
                    raise TranspilerError(
                        "HighLevelSynthesis failed to synthesize the control modifier."
                    )

            elif isinstance(modifier, PowerModifier):
                # QuantumCircuit has power method, and Gate needs to be converted
                # to a quantum circuit.
                if not isinstance(synthesized, QuantumCircuit):
                    synthesized = instruction_to_circuit(synthesized)

                synthesized = synthesized.power(modifier.power)

            else:
                raise TranspilerError(f"Unknown modifier {modifier}.")

        return synthesized

    def _definitely_skip_node(self, node: DAGOpNode, qubits: tuple[int] | None) -> bool:
        """Fast-path determination of whether a node can certainly be skipped (i.e. nothing will
        attempt to synthesise it) without accessing its Python-space `Operation`.

        This is tightly coupled to `_recursively_handle_op`; it exists as a temporary measure to
        avoid Python-space `Operation` creation from a `DAGOpNode` if we wouldn't do anything to the
        node (which is _most_ nodes)."""
        return (
            # The fast path is just for Rust-space standard gates (which excludes
            # `AnnotatedOperation`).
            node.is_standard_gate()
            # If it's a controlled gate, we might choose to do funny things to it.
            and not node.is_controlled_gate()
            # If there are plugins to try, they need to be tried.
            and not self._methods_to_try(node.name)
            # If all the above constraints hold, and it's already supported or the basis translator
            # can handle it, we'll leave it be.
            and (
                self._instruction_supported(node.name, qubits)
                # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
                # `BasisTranslator`, and this is supposed to just be temporary til this is moved
                # into Rust space.
                or (
                    self._equiv_lib is not None
                    and equivalence.Key(name=node.name, num_qubits=node.num_qubits)
                    in self._equiv_lib._key_to_node_index
                )
            )
        )

    def _instruction_supported(self, name: str, qubits: tuple[int] | None) -> bool:
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if self._target is None or self._target.num_qubits is None:
            return name in self._device_insts
        return self._target.instruction_supported(operation_name=name, qargs=qubits)


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin.

    For N <= 3 qubits this is the optimal CX cost decomposition by Bravyi, Maslov.
    For N > 3 qubits this is done using the general non-optimal greedy compilation
    routine from reference by Bravyi, Hu, Maslov, Shaydulin.

    This plugin name is :``clifford.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_ag(high_level_object)
        return decomposition


class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None
        return decomposition


class GreedySynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the greedy synthesis
    Bravyi-Hu-Maslov-Shaydulin method.

    This plugin name is :``clifford.greedy`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_greedy(high_level_object)
        return decomposition


class LayerSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers.

    This plugin name is :``clifford.layers`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_layers(high_level_object)
        return decomposition


class LayerLnnSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers, with each layer synthesized
    adhering to LNN connectivity.

    This plugin name is :``clifford.lnn`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method.

    This plugin name is :``linear_function.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""

        if not isinstance(high_level_object, LinearFunction):
            raise TranspilerError(
                "PMHSynthesisLinearFunction only accepts objects of type LinearFunction"
            )

        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_depth_line_kms(mat)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method.

    This plugin name is :``linear_function.pmh`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * section size: The size of each section used in the Patel–Markov–Hayes algorithm [1].
    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""

        if not isinstance(high_level_object, LinearFunction):
            raise TranspilerError(
                "PMHSynthesisLinearFunction only accepts objects of type LinearFunction"
            )

        section_size = options.get("section_size", 2)
        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_count_full_pmh(mat, section_size=section_size)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.

    This plugin name is :``permutation.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition


class QFTSynthesisFull(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using all-to-all connectivity.

    This plugin name is :``qft.full`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): The degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].
    * insert_barriers (bool): If True, barriers are inserted as visualization improvement.
    * inverse (bool): If True, the inverse Fourier transform is constructed.
    * name (str): The name of the circuit.

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""
        if not isinstance(high_level_object, QFTGate):
            raise TranspilerError(
                "The synthesis plugin 'qft.full` only applies to objects of type QFTGate."
            )

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)
        insert_barriers = options.get("insert_barriers", False)
        inverse = options.get("inverse", False)
        name = options.get("name", None)

        decomposition = synth_qft_full(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
            insert_barriers=insert_barriers,
            inverse=inverse,
            name=name,
        )
        return decomposition


class QFTSynthesisLine(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using linear connectivity.

    This plugin name is :``qft.line`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): the degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""
        if not isinstance(high_level_object, QFTGate):
            raise TranspilerError(
                "The synthesis plugin 'qft.line` only applies to objects of type QFTGate."
            )

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)

        decomposition = synth_qft_line(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
        )
        return decomposition


class TokenSwapperSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the token swapper algorithm.

    This plugin name is :``permutation.token_swapper`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    In more detail, this plugin is used to synthesize objects of type `PermutationGate`.
    When synthesis succeeds, the plugin outputs a quantum circuit consisting only of swap
    gates. When synthesis does not succeed, the plugin outputs `None`.

    If either `coupling_map` or `qubits` is None, then the synthesized circuit
    is not required to adhere to connectivity constraints, as is the case
    when the synthesis is done before layout/routing.

    On the other hand, if both `coupling_map` and `qubits` are specified, the synthesized
    circuit is supposed to adhere to connectivity constraints. At the moment, the
    plugin only creates swap gates between qubits in `qubits`, i.e. it does not use
    any other qubits in the coupling map (if such synthesis is not possible, the
    plugin  outputs `None`).

    The plugin supports the following plugin-specific options:

    * trials: The number of trials for the token swapper to perform the mapping. The
      circuit with the smallest number of SWAPs is returned.
    * seed: The argument to the token swapper specifying the seed for random trials.
    * parallel_threshold: The argument to the token swapper specifying the number of nodes
      in the graph beyond which the algorithm will use parallel processing.

    For more details on the token swapper algorithm, see to the paper:
    `arXiv:1902.09102 <https://arxiv.org/abs/1902.09102>`__.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""

        trials = options.get("trials", 5)
        seed = options.get("seed", 0)
        parallel_threshold = options.get("parallel_threshold", 50)

        pattern = high_level_object.pattern
        pattern_as_dict = {j: i for i, j in enumerate(pattern)}

        # When the plugin is called from the HighLevelSynthesis transpiler pass,
        # the coupling map already takes target into account.
        if coupling_map is None or qubits is None:
            # The abstract synthesis uses a fully connected coupling map, allowing
            # arbitrary connections between qubits.
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            # The concrete synthesis uses the coupling map restricted to the set of
            # qubits over which the permutation gate is defined. If we allow using other
            # qubits in the coupling map, replacing the node in the DAGCircuit that
            # defines this PermutationGate by the DAG corresponding to the constructed
            # decomposition becomes problematic. Note that we allow the reduced
            # coupling map to be disconnected.
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        graph = used_coupling_map.graph.to_undirected()
        swapper = ApproximateTokenSwapper(graph, seed=seed)

        try:
            swapper_result = swapper.map(
                pattern_as_dict, trials, parallel_threshold=parallel_threshold
            )
        except rx.InvalidMapping:
            swapper_result = None

        if swapper_result is not None:
            decomposition = QuantumCircuit(len(graph.node_indices()))
            for swap in swapper_result:
                decomposition.swap(*swap)
            return decomposition

        return None
