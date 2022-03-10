# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager for optimization level 2, providing medium optimization.

Level 2 pass manager: medium optimization by noise adaptive qubit mapping and
gate cancellation using commutativity rules.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler.passes import ASAPSchedule
from qiskit.transpiler.passes import AlignMeasures
from qiskit.transpiler.passes import ValidatePulseGates
from qiskit.transpiler.passes import PulseGates
from qiskit.transpiler.passes import PadDelay
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import ContainsInstruction
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

from qiskit.transpiler import TranspilerError


def level_2_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 2 pass manager: medium optimization by initial layout selection and
    gate cancellation using commutativity rules.

    This pass manager applies the user-given initial layout. If none is given, a search
    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.
    If no such layout is found, qubits are laid out on the most densely connected subset
    which also exhibits the best gate fidelities.

    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation and redundant
    reset removal are performed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 2 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target

    # 1. Unroll to 1q or 2q gates
    _unroll3q = [
        # Use unitary synthesis for basis aware decomposition of UnitaryGates
        UnitarySynthesis(
            basis_gates,
            approximation_degree=approximation_degree,
            method=unitary_synthesis_method,
            min_qubits=3,
            plugin_config=unitary_synthesis_plugin_config,
        ),
        Unroll3qOrMore(),
    ]

    # 2. Search for a perfect layout, or choose a dense layout, if no layout given
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        # layout hasn't been set yet
        return not property_set["layout"]

    def _vf2_match_not_found(property_set):
        # If a layout hasn't been set by the time we run vf2 layout we need to
        # run layout
        if property_set["layout"] is None:
            return True
        # if VF2 layout stopped for any reason other than solution found we need
        # to run layout since VF2 didn't converge.
        if (
            property_set["VF2Layout_stop_reason"] is not None
            and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND
        ):
            return True
        return False

    # 2a. Try using VF2 layout to find a perfect layout
    _choose_layout_0 = (
        []
        if pass_manager_config.layout_method
        else VF2Layout(
            coupling_map,
            seed=seed_transpiler,
            call_limit=int(5e6),  # Set call limit to ~10 sec with retworkx 0.10.2
            time_limit=10.0,
            properties=backend_properties,
        )
    )

    # 2b. if VF2 layout doesn't converge on a solution use layout_method (dense) to get a layout
    if layout_method == "trivial":
        _choose_layout_1 = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _choose_layout_1 = DenseLayout(coupling_map, backend_properties)
    elif layout_method == "noise_adaptive":
        _choose_layout_1 = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _choose_layout_1 = SabreLayout(coupling_map, max_iterations=2, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    # 3. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_method == "basic":
        _swap += [BasicSwap(coupling_map)]
    elif routing_method == "stochastic":
        _swap += [StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)]
    elif routing_method == "lookahead":
        _swap += [LookaheadSwap(coupling_map, search_depth=5, search_width=5)]
    elif routing_method == "sabre":
        _swap += [SabreSwap(coupling_map, heuristic="decay", seed=seed_transpiler)]
    elif routing_method == "none":
        _swap += [
            Error(
                msg=(
                    "No routing method selected, but circuit is not routed to device. "
                    "CheckMap Error: {check_map_msg}"
                ),
                action="raise",
            )
        ]
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # 5. Unroll to the basis
    if translation_method == "unroller":
        _unroll = [Unroller(basis_gates)]
    elif translation_method == "translator":
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        _unroll = [
            # Use unitary synthesis for basis aware decomposition of UnitaryGates before
            # custom unrolling
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_plugin_config,
            ),
            UnrollCustomDefinitions(sel, basis_gates),
            BasisTranslator(sel, basis_gates, target),
        ]
    elif translation_method == "synthesis":
        _unroll = [
            # Use unitary synthesis for basis aware decomposition of UnitaryGates before
            # collection
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_plugin_config,
                min_qubits=3,
            ),
            Unroll3qOrMore(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates),
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_plugin_config,
            ),
        ]
    else:
        raise TranspilerError("Invalid translation method %s." % translation_method)

    # 6. Fix any bad CX directions
    _direction_check = [CheckGateDirection(coupling_map, target)]

    def _direction_condition(property_set):
        return not property_set["is_direction_mapped"]

    _direction = [GateDirection(coupling_map, target)]

    # 7. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    # 8. 1q rotation merge and commutative cancellation iteratively until no more change in depth
    _depth_check = [Depth(), FixedPoint("depth")]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    _opt = [
        Optimize1qGatesDecomposition(basis_gates),
        CommutativeCancellation(basis_gates=basis_gates),
    ]

    # 9. Unify all durations (either SI, or convert to dt if known)
    # Schedule the circuit only when scheduling_method is supplied
    _time_unit_setup = [ContainsInstruction("delay")]
    _time_unit_conversion = [TimeUnitConversion(instruction_durations)]

    def _contains_delay(property_set):
        return property_set["contains_delay"]

    _scheduling = []
    if scheduling_method:
        _scheduling += _time_unit_conversion
        if scheduling_method in {"alap", "as_late_as_possible"}:
            _scheduling += [ALAPSchedule(instruction_durations), PadDelay()]
        elif scheduling_method in {"asap", "as_soon_as_possible"}:
            _scheduling += [ASAPSchedule(instruction_durations), PadDelay()]
        else:
            raise TranspilerError("Invalid scheduling method %s." % scheduling_method)

    # 10. Call measure alignment. Should come after scheduling.
    if (
        timing_constraints.granularity != 1
        or timing_constraints.min_length != 1
        or timing_constraints.acquire_alignment != 1
    ):
        _alignments = [
            ValidatePulseGates(
                granularity=timing_constraints.granularity, min_length=timing_constraints.min_length
            ),
            AlignMeasures(alignment=timing_constraints.acquire_alignment),
        ]
    else:
        _alignments = []

    # Build pass manager
    pm2 = PassManager()
    if coupling_map or initial_layout:
        pm2.append(_given_layout)
        pm2.append(_unroll3q)
        pm2.append(_choose_layout_0, condition=_choose_layout_condition)
        pm2.append(_choose_layout_1, condition=_vf2_match_not_found)
        pm2.append(_embed)
        pm2.append(_swap_check)
        pm2.append(_swap, condition=_swap_condition)
    pm2.append(_unroll)
    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pm2.append(_direction_check)
        pm2.append(_direction, condition=_direction_condition)
    pm2.append(_reset)
    pm2.append(_depth_check + _opt + _unroll, do_while=_opt_control)
    if inst_map and inst_map.has_custom_gate():
        pm2.append(PulseGates(inst_map=inst_map))
    if scheduling_method:
        pm2.append(_scheduling)
    elif instruction_durations:
        pm2.append(_time_unit_setup)
        pm2.append(_time_unit_conversion, condition=_contains_delay)
    pm2.append(_alignments)
    return pm2
