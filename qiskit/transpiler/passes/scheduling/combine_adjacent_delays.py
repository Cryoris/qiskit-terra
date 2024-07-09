# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Combine adjacent delays into multi-delay instructions."""

from __future__ import annotations
from enum import Enum

import logging

from pprint import pformat
from dataclasses import dataclass
from collections import defaultdict, Counter

from qiskit.circuit import Delay, QuantumCircuit, Qubit, Instruction
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import TransformationPass, Target

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Delay event type, which is either begin or end."""

    BEGIN = 0
    END = 1


@dataclass
class DelayEvent:
    """Represent a single delay event."""

    type: EventType  # One of 'begin' or 'end'  TODO could be an Enum
    time: int  # Time in thi.e.e circuit, in dt
    op_node: DAGOpNode  # The node for the circuit delay

    @property
    def qargs(self) -> set[Qubit]:
        """Qubits the delay is acting on. Not sorted."""
        return set(self.op_node.qargs)

    @staticmethod
    def sort_key(event: DelayEvent) -> tuple(int, int):
        """Sort events, first by time then by type ('end' events come for 'begin' events)."""
        return (
            event.time,  # Sort by event time
            0 if event.type == EventType.END else 1,  # With 'end' events before 'begin'
        )


@dataclass
class AdjacentDelayBlock:
    """Group of circuit delays which are collectively adjacent in time and on device."""

    events: list[DelayEvent]
    active_qubits: set[Qubit]

    def validate(self) -> None:
        """Validate the list of delay events in the adjacent block.

        Raises:
            RuntimeError: If the blocks are not ordered by time and event type.
        """

        for idx, event in enumerate(self.events[:-1]):
            if event.time > self.events[idx + 1].time:
                raise RuntimeError("adjacent_delay_block.events not ordered by time")

            if event.time == self.events[idx + 1].time:
                # At same time, can either be ('begin', 'begin'), ('end', 'begin') or ('end', 'end')
                if (event.type, self.events[idx + 1].type) == (EventType.BEGIN, EventType.END):
                    raise RuntimeError(
                        "Events in the AdjacentDelayBlock are not correctly sorted by "
                        "event type. At same time, we can have either of (begin, begin), "
                        f"(end, begin) or (end, end). This happened at time {event.time}."
                    )


# TODO: Is ConcurrentDelayEvent better? "Grouped" does not imply same time
# TODO: Why have an AdjacentDelayBlock and a separate GroupedDelayEvent? Don't they refer to the same?
@dataclass
class GroupedDelayEvent:
    """Grouped event of single type and time, but spanning one or more qubits/Delay instructions."""

    type: EventType
    time: int
    op_nodes: list[DAGOpNode]

    def __init__(self, event_type: EventType, time: int, op_nodes: list[DAGOpNode]) -> None:
        self.type = event_type
        self.time = time
        self.op_nodes = op_nodes
        self._qargs = set(qarg for op_node in self.op_nodes for qarg in op_node.qargs)

    @property
    def qargs(self) -> set[Qubit]:
        """The qubits involved in the grouped delay event."""
        # return self._qargs
        return set(qarg for op_node in self.op_nodes for qarg in op_node.qargs)

    def __repr__(self) -> str:
        return (
            f"GroupedDelayEvent({self.type}),\n"
            f"                  {self.time}),\n"
            f"                  {[(op.op.duration, [op.qargs]) for op in self.op_nodes]})\n"
        )


# TODO: If they are adjacent, why aren't they grouped?
@dataclass
class AdjacentGroupedDelayBlock:
    """Collection of circuit delay groups which are collectively adjacent in time and on device."""

    events: list[GroupedDelayEvent]

    def validate(self) -> None:
        """Validate the order of the grouped delay events."""
        # events in order start end, type
        last_time = -1
        last_type = None
        for grouped_delay_event in self.events:
            if grouped_delay_event.time > last_time:
                last_time = grouped_delay_event.time
                last_type = grouped_delay_event.type
                continue

            if grouped_delay_event.time < last_time:
                raise RuntimeError("out of time order")

            if (last_type, grouped_delay_event.type) != (EventType.END, EventType.BEGIN):
                raise RuntimeError("Event types are out of order, found (end, begin).")

        # every qubit has matching start/end pairs
        all_qubits_event_types = defaultdict(list)
        for event in self.events:
            for qarg in event.qargs:
                all_qubits_event_types[qarg].append(event.type)

        for qubit_event_types in all_qubits_event_types.values():
            if len(qubit_event_types) % 2 != 0:
                raise RuntimeError(f"Odd number of qubit events: {len(qubit_event_types)}.")

            if qubit_event_types != [EventType.BEGIN, EventType.END] * int(
                len(qubit_event_types) / 2
            ):
                raise RuntimeError("Qubit events do not strictly follow open/close/open/close/...")


# TODO: why do we have this on top of the AdjacentGroupedDelayBlock?
@dataclass
class CollectedJointDelay:
    """Collection of Delay ops from input circuit to be output as a joint delay."""

    num_qubits: int
    begin_delay_event: GroupedDelayEvent
    end_delay_event: GroupedDelayEvent
    open_op_nodes: list[
        DAGOpNode
    ]  # N.B. These op_nodes in general will extend beyond {begin,end}_delay_event.

    def __iter__(self):
        return iter(
            (self.num_qubits, self.begin_delay_event, self.end_delay_event, self.open_op_nodes)
        )


class MultiDelay(Instruction):
    """A multi-qubit delay instruction."""

    def __init__(self, num_qubits: int, duration: int) -> None:
        """
        Args:
            num_qubits: The number of qubits the delay acts on.
            duration: The delay duration in terms of the target's ``dt``.
        """
        super().__init__("multi_delay", num_qubits, 0, [], duration=duration)

    def _define(self) -> None:
        # This could be defined in terms of Delay instructions, but we are only
        # using this as placeholder for the DynamicalDecouplingMulti pass, so it should
        # never be unrolled.
        return None


# TODO: do we need this?
@dataclass
class ReplacementDelay:
    """Used to replace a set of DAG nodes with a MultiDelay or Delay."""

    new_delay_op: Delay | MultiDelay
    start_time: int
    end_time: int
    replacing_delay_nodes: list[
        DAGOpNode
    ]  # List of DelayOpNodes on the original DAG which overlap with this new delay.


class CombineAdjacentDelays(TransformationPass):
    """Combine :class:`.Delay` instructions on adjacent qubits.

    TODO: This should update the property_set["node_start_time"] to avoid having to call
          a schedule analysis after ths pass
    TODO: Whatever object ends up in the circuit should be API-documented.
    """

    def __init__(self, target: Target, min_joinable_delay_duration: float = 0.0) -> None:
        """
        Args:
            target: The :class:`.Target` of the circuit.
            min_joinable_delay_duration: Do not join :class:`.Delay` instruction shorter than
                this time.
        """
        super().__init__()

        self.cmap = target.build_coupling_map()  # TODO: does this need to be public?
        self._min_joinable_delay_duration = min_joinable_delay_duration

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        bit_idx_locs = {bit: idx for idx, bit in enumerate(dag.qubits)}

        # Find and sort every leading/trailing edge of a delay in the circuit.
        # These will be the places we'll examine to split/combine delay ops.
        sorted_delay_events = sorted(
            (
                DelayEvent(
                    event_type,
                    (
                        start_time
                        if event_type == EventType.BEGIN
                        else start_time + op_node.op.duration
                    ),
                    op_node,
                )
                for op_node, start_time in self.property_set["node_start_time"].items()
                if (
                    op_node.op.name == "delay"
                    and start_time
                    != 0  # Skip delays at start of circuit  # TODO: should this be an argument?
                    # and start_time + op_node.op.duration < dag.duration  # Skip delays at end of circuit  # TODO: should this be an argument?
                    and op_node.op.duration > self._min_joinable_delay_duration
                )
                for event_type in (EventType.BEGIN, EventType.END)
            ),
            key=DelayEvent.sort_key,
        )

        # TODO: is pformat supported per default?
        logger.debug("sorted_delay_events: %s", pformat(sorted_delay_events))

        # Collect grouped delays by concurrency in time and adjacency on device.
        adjacent_delay_blocks = _collect_adjacent_delay_blocks(
            sorted_delay_events, self.cmap, bit_idx_locs
        )

        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below, so validate.
        # TODO: it seems validation better happen after the collection, not manually later
        for adjacent_delay_block in adjacent_delay_blocks:
            adjacent_delay_block.validate()

        # Within each adjacent delay block, collect events by time, type.
        # This will faciltate knowing where to split them into joint N-qubit delays.
        grouped_delay_blocks = _group_delay_blocks(adjacent_delay_blocks)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                pformat(
                    (
                        "---post_group_by_time/event---",
                        ("grouped_delays_blocks: ", grouped_delay_blocks),
                        ("len(grouped_delay_blocks): ", len(grouped_delay_blocks)),
                    )
                )
            )

        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below.
        # So verify.
        # TODO: see above, this should happen automatically IMO
        for grouped_delay_block in grouped_delay_blocks:
            grouped_delay_block.validate()

        logger.info("Begin finding and isolating widest delays within groups.")

        # Walk through existing delay blocks, finding interval where the most adjacent qubits are
        # joinly idle. If that interval is longer than self._min_joinable_delay_duration, combine
        # events which begin and end it into a new AdjacentGroupedDelayBlock and add to output.
        # Split remainder of original delay block into two new delay blocks and add those onto
        # queue to be further split if possible.

        collected_joint_delays = _subdivide_grouped_delays(
            grouped_delay_blocks,
            self._min_joinable_delay_duration,
            self.cmap,
            bit_idx_locs,
            self.property_set,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(pformat(("collected_joint_delays", collected_joint_delays)))

        # Create new N-Qubit delay ops for joint delays.

        replacement_delays = []
        for widest in collected_joint_delays:
            start_time = widest.begin_delay_event.time
            end_time = widest.end_delay_event.time

            replacement = ReplacementDelay(
                MultiDelay(widest.num_qubits, end_time - start_time),
                start_time,
                end_time,
                widest.open_op_nodes,
            )

            replacement_delays.append(replacement)

        # Replacement strategy:
        # 1) Split each doomed delay in to N 1q placeholder delays with duration of multi-qubit delays
        # 2) Combine blocks of new multi-qubit delays

        # Have mapping from new joint delay ops to original delays they replace, but
        # need inverse mapping (original delay to replacements) to know lengths of
        # placeholder delays.
        doomed_delay_to_replacements = defaultdict(
            list
        )  # existing_delay_node, [sorted_list_of_replacement_delays]
        for replacement_delay in sorted(
            replacement_delays, key=lambda k: (k.start_time, k.end_time)
        ):  # Might already be in right order?

            for doomed_delay_node in replacement_delay.replacing_delay_nodes:
                doomed_delay_to_replacements[doomed_delay_node].append(replacement_delay)

        # Need a map from each new delay to the corresponding placeholders to know which to replace.
        # (Order doesn't matter.)
        replacement_to_placeholder = defaultdict(list)

        for doomed_delay_node, replacements in doomed_delay_to_replacements.items():
            placeholder_delay = QuantumCircuit(1)
            for replacement in replacements:
                placeholder_delay.delay(replacement.new_delay_op.duration, 0)

            placeholder_dag = circuit_to_dag(placeholder_delay)
            placeholder_delay_node_ids = [
                node._node_id for node in placeholder_dag.topological_op_nodes()
            ]

            out_node_map = dag.substitute_node_with_dag(doomed_delay_node, placeholder_dag)
            placeholder_delay_nodes = [
                out_node_map[node_id] for node_id in placeholder_delay_node_ids
            ]
            self.property_set["node_start_time"].pop(doomed_delay_node)

            for replacement, placeholder_delay_node in zip(replacements, placeholder_delay_nodes):
                replacement_to_placeholder[id(replacement.new_delay_op)].append(
                    placeholder_delay_node
                )

        for replacement in replacement_delays:
            placeholder_nodes = replacement_to_placeholder[id(replacement.new_delay_op)]
            new_node = dag.replace_block_with_op(
                placeholder_nodes,
                replacement.new_delay_op,
                {node.qargs[0]: idx for idx, node in enumerate(placeholder_nodes)},
                cycle_check=True,
            )
            self.property_set["node_start_time"][new_node] = replacement.start_time

        for delay_node in dag.named_nodes("delay"):
            if delay_node.op.num_qubits == 1:
                continue

            if (
                delay_node.op.num_qubits > 1
                and delay_node.op.duration < self._min_joinable_delay_duration
            ):
                raise RuntimeError("")

            # len(subgraph) == num_qubits
            if not all(
                any(
                    self.cmap.distance(bit_idx_locs[test_qubit], bit_idx_locs[op_qubit]) == 1
                    for op_qubit in delay_node.qargs
                )
                for test_qubit in delay_node.qargs
            ):
                raise RuntimeError([bit_idx_locs[q] for q in delay_node.qargs])

        return dag


def _collect_adjacent_delay_blocks(sorted_delay_events, cmap, bit_idx_locs):
    open_delay_blocks = []
    closed_delay_blocks = []

    def _open_delay_block(delay_event):
        open_delay_blocks.append(
            AdjacentDelayBlock(events=[delay_event], active_qubits=set(delay_event.qargs))
        )
        return open_delay_blocks[-1]

    def _update_delay_block(open_delay_block, delay_event):
        """Add another delay event to an existing block to either extend or close it."""
        open_delay_block.events.append(delay_event)

        if delay_event.type == EventType.BEGIN:  # TODO: this is never called
            open_delay_block.active_qubits += set(delay_event.qargs)
        elif delay_event.type == EventType.END:
            open_delay_block.active_qubits -= set(delay_event.qargs)
        else:
            raise RuntimeError("bad event type")

        if not open_delay_block.active_qubits:
            open_delay_blocks.remove(open_delay_block)
            closed_delay_blocks.append(open_delay_block)

    def _combine_delay_blocks(delay_blocks):
        survivor, *doomed = delay_blocks

        # TODO: this seems to re-iterated over doomed blocks multiple times
        for doomed_delay_group in doomed:
            # Add events and qubits from doomed block to survivor.
            if survivor.active_qubits.intersection(doomed_delay_group.active_qubits):
                raise RuntimeError("More than one open delay on a qubit?")

            survivor.events.extend(doomed_delay_group.events)
            survivor.active_qubits.update(doomed_delay_group.active_qubits)

            open_delay_blocks.remove(doomed_delay_group)
            survivor.events.sort(key=DelayEvent.sort_key)  # Maintain sorted event order

    for delay_event in sorted_delay_events:
        # This could be avoided by keeping a map of device qubit to open
        # block and only considering neighbors of current event.
        adjacent_open_delay_blocks = [
            open_delay
            for open_delay in open_delay_blocks
            if any(
                cmap.distance(  # TODO: use cmap.neighbors?
                    bit_idx_locs[delay_event.op_node.qargs[0]], bit_idx_locs[open_delay_qubit]
                )
                <= 1
                for open_delay_qubit in open_delay.active_qubits
            )
        ]

        if delay_event.type == EventType.BEGIN:
            # If crossing a begin edge, check if there are any open delays that are adjacent.
            # If so, add current event to that group.

            if len(adjacent_open_delay_blocks) == 0:
                # Make a new delay
                _open_delay_block(delay_event)
            else:
                # Make a new block and combine that with adjacent open blocks
                new_block = _open_delay_block(delay_event)
                _combine_delay_blocks(adjacent_open_delay_blocks + [new_block])

        if delay_event.type == EventType.END:
            # If crossing a end edge, remove this qubit from the actively delaying qubits"
            if len(adjacent_open_delay_blocks) != 1:
                raise RuntimeError("Closing edge w/o an open delay?")

            _update_delay_block(adjacent_open_delay_blocks[0], delay_event)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                pformat(
                    (
                        "---exit---",
                        delay_event,
                        ("open_delays: ", open_delay_blocks),
                        ("closed_delays: ", closed_delay_blocks),
                    )
                )
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            pformat(
                (
                    "---post_collect---",
                    ("open_delays: ", open_delay_blocks),
                    ("len(closed_delays): ", len(closed_delay_blocks)),
                )
            )
        )

    if open_delay_blocks:
        raise RuntimeError("Failed to close all open delays.")
    for closed_delay in closed_delay_blocks:
        if closed_delay.active_qubits:
            raise RuntimeError("Failed remove active qubits on closed delays.")

    return closed_delay_blocks


def _group_delay_blocks(adjacent_delay_blocks):
    # TODO these functions are complex enough to be properly documented
    grouped_delay_blocks = []

    for adjacent_delay_block in adjacent_delay_blocks:
        grouped_delay_block = AdjacentGroupedDelayBlock([])

        for delay_event in adjacent_delay_block.events:
            # If we're on the first edge, it gets a pass.
            if not grouped_delay_block.events:
                grouped_delay_block.events.append(
                    GroupedDelayEvent(delay_event.type, delay_event.time, [delay_event.op_node])
                )
                continue

            # Look at last known grouped event.
            newest_grouped_delay = grouped_delay_block.events[-1]

            # If newest grouped event is earlier than current, need a new group.
            if delay_event.time > newest_grouped_delay.time:
                grouped_delay_block.events.append(
                    GroupedDelayEvent(delay_event.type, delay_event.time, [delay_event.op_node])
                )

                continue

            # If newest grouped event is _later_ than current, question our worldview.
            if delay_event.time < newest_grouped_delay.time:
                raise RuntimeError("Did not find a matching event group.")

            # If newest grouped has same time and type, add current event to that group.
            if delay_event.type == newest_grouped_delay.type:
                newest_grouped_delay.op_nodes.append(delay_event.op_node)
                continue

            # If we've passed all the above, have same time but different type, so add a new group.
            grouped_delay_block.events.append(
                GroupedDelayEvent(delay_event.type, delay_event.time, [delay_event.op_node])
            )

        grouped_delay_block.validate()
        grouped_delay_blocks.append(grouped_delay_block)

    return grouped_delay_blocks


def _subdivide_grouped_delays(
    grouped_delay_blocks, min_joinable_delay_duration, cmap, bit_idx_locs, property_set
):
    closed_joint_delays = []

    while grouped_delay_blocks:
        logger.info(
            "In _subdivide_grouped_delays loop: %s done, %s remain.",
            len(closed_joint_delays),
            len(grouped_delay_blocks),
        )
        grouped_delay = grouped_delay_blocks.pop(0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Examining grouped_delay: %s", pformat(grouped_delay))

        widest = _find_widest_connected_joint_delay(grouped_delay)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Found widest interval: %s", widest)

        new_delay_width, new_delay_begin, new_delay_end, replacing_op_nodes = widest

        if new_delay_width == 1:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Widest segment has width 1. Adding to closed delays.")
            closed_joint_delays.append(widest)

            if len(grouped_delay.events) > 2:
                raise RuntimeError("widest[0] == 1 but more than one open/close pair")
        elif new_delay_end is None:
            raise RuntimeError(f"Widest was never closed?: {widest}")
        else:
            new_delay_duration = new_delay_end.time - new_delay_begin.time

            split_left_idx = grouped_delay.events.index(new_delay_begin)
            split_right_idx = grouped_delay.events.index(new_delay_end)

            if split_right_idx != split_left_idx + 1:
                raise RuntimeError("widest begin and end non-adjacent")

            if new_delay_duration >= min_joinable_delay_duration:
                closed_joint_delays.append(widest)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Widest exceeds min_joinable_delay_duration, "
                        "splitting group_delay_events at index "
                        "%s and %s.",
                        split_left_idx,
                        split_right_idx,
                    )

                # N.B. These won't be valid until after they have the following events added.

                if split_left_idx > 0:
                    new_left_delay_group = AdjacentGroupedDelayBlock(
                        grouped_delay.events[:split_left_idx]
                    )

                    new_left_delay_group.events.append(
                        GroupedDelayEvent(
                            event_type=EventType.END,
                            time=new_delay_begin.time,
                            op_nodes=[
                                op_node
                                for op_node in replacing_op_nodes
                                if op_node not in new_delay_begin.op_nodes
                            ],
                        )
                    )

                    new_left_ungroup = _ungroup_adj_grouped_delay_block(new_left_delay_group)
                    left_adjacent_delay_blocks = _collect_adjacent_delay_blocks(
                        new_left_ungroup.events, cmap, bit_idx_locs
                    )
                    left_grouped_delay_blocks = _group_delay_blocks(left_adjacent_delay_blocks)

                    # if logger.isEnabledFor(logging.DEBUG):
                    #     logger.debug('Creating group from left-of widest: '
                    #                  f'{pformat(new_left_delay_group)}')
                    # if not new_left_delay_group:
                    #     import pdb; pdb.set_trace()
                    # grouped_delay_blocks.append(new_left_delay_group.validate())

                    grouped_delay_blocks.extend(left_grouped_delay_blocks)

                if split_right_idx < len(grouped_delay.events) - 1:
                    # KDK Need to check here that remaining block is still adjacent, else make more than one

                    new_right_delay_group = AdjacentGroupedDelayBlock(
                        grouped_delay.events[split_right_idx + 1 :]
                    )

                    new_right_delay_group.events.insert(
                        0,
                        GroupedDelayEvent(
                            event_type=EventType.BEGIN,
                            time=new_delay_end.time,
                            op_nodes=[
                                op_node
                                for op_node in replacing_op_nodes
                                if op_node not in new_delay_end.op_nodes
                            ],
                        ),
                    )

                    new_right_ungroup = _ungroup_adj_grouped_delay_block(new_right_delay_group)
                    right_adjacent_delay_blocks = _collect_adjacent_delay_blocks(
                        new_right_ungroup.events, cmap, bit_idx_locs
                    )
                    right_grouped_delay_blocks = _group_delay_blocks(right_adjacent_delay_blocks)

                    # if logger.isEnabledFor(logging.DEBUG):
                    #     logger.debug('Creating group from right-of widest: '
                    #                  f'{pformat(new_right_delay_group)}')
                    # if not new_right_delay_group:
                    #     import pdb; pdb.set_trace()
                    # grouped_delay_blocks.append(new_right_delay_group.validate())
                    grouped_delay_blocks.extend(right_grouped_delay_blocks)

            else:
                # If new_delay_duration is shorter than min_joinable_delay_duration,
                # attempt to split current group into two grouped delays which will
                # admit the widest possible delay when they are each processed.

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Widest segment duration %s shorter than min_joinable_delay_duration %s. "
                        "Splitting group_delay_events at index %s and %s.",
                        new_delay_duration,
                        min_joinable_delay_duration,
                        split_left_idx,
                        split_right_idx,
                    )

                # If there are qubits which both opened and closed this delay group,
                # they will have a delay < min_joinable_delay_duration, so pull them out of
                # this group and give them all 1q Delays.
                narrow_single_delay_qubits = new_delay_begin.qargs.intersection(new_delay_end.qargs)

                for single_delay_qubit in narrow_single_delay_qubits:
                    narrow_single_delay_op_nodes = [
                        op_node
                        for op_node in replacing_op_nodes
                        if single_delay_qubit in op_node.qargs
                    ]

                    single_delay_begin = GroupedDelayEvent(
                        event_type=EventType.BEGIN,
                        time=new_delay_begin.time,
                        op_nodes=narrow_single_delay_op_nodes,
                    )

                    single_delay_end = GroupedDelayEvent(
                        event_type=EventType.END,
                        time=new_delay_end.time,
                        op_nodes=narrow_single_delay_op_nodes,
                    )

                    narrow_single_delay = CollectedJointDelay(
                        num_qubits=1,
                        begin_delay_event=single_delay_begin,
                        end_delay_event=single_delay_end,
                        open_op_nodes=narrow_single_delay_op_nodes,
                    )

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Adding single-qubit delays on splintered qubit. %s",
                            pformat(narrow_single_delay),
                        )

                    closed_joint_delays.append(narrow_single_delay)

                if narrow_single_delay_qubits:

                    # If there were qubits in new_delay_begin/new_delay_end that did not get cut out
                    # into singles retain their begin/end events, but otherwise retain as one group.
                    residual_begin_qubits = new_delay_begin.qargs - narrow_single_delay_qubits
                    residual_end_qubits = new_delay_end.qargs - narrow_single_delay_qubits

                    if residual_end_qubits:
                        residual_end_event = GroupedDelayEvent(
                            event_type=EventType.END,
                            time=new_delay_end.time,
                            op_nodes=[
                                op_node
                                for op_node in replacing_op_nodes
                                if residual_end_qubits.intersection(op_node.qargs)
                            ],
                        )

                        grouped_delay.events[split_right_idx] = residual_end_event
                    else:
                        grouped_delay.events.pop(split_right_idx)

                    if residual_begin_qubits:
                        residual_begin_event = GroupedDelayEvent(
                            event_type=EventType.BEGIN,
                            time=new_delay_begin.time,
                            op_nodes=[
                                op_node
                                for op_node in replacing_op_nodes
                                if residual_begin_qubits.intersection(op_node.qargs)
                            ],
                        )

                        grouped_delay.events[split_left_idx] = residual_begin_event
                    else:
                        grouped_delay.events.pop(split_left_idx)

                    # Do not want to split in this case, we could e.g. have a delay that started on
                    # LHS and ends on RHS, and we don't want to pull forward either.
                    new_resid_ungroup = _ungroup_adj_grouped_delay_block(grouped_delay)
                    resid_adjacent_delay_blocks = _collect_adjacent_delay_blocks(
                        new_resid_ungroup.events, cmap, bit_idx_locs
                    )
                    resid_grouped_delay_blocks = _group_delay_blocks(resid_adjacent_delay_blocks)

                    grouped_delay_blocks.extend(resid_grouped_delay_blocks)
                else:
                    # If we didn't have any narrow 1q delays to remove, need to partion existing
                    # qubits with condition that new_delay_begin.qargs and new_delay_end.qargs end
                    # up in different partitions.

                    begin_event_partition = set(new_delay_begin.op_nodes)
                    end_event_partition = set(new_delay_end.op_nodes)
                    solo_partitions = []

                    unassigned_op_nodes = set(
                        op_node for event in grouped_delay.events for op_node in event.op_nodes
                    )
                    unassigned_op_nodes -= begin_event_partition
                    unassigned_op_nodes -= end_event_partition

                    # N.B. Will need to re-walk/sort, re-group

                    for unassigned_op_node in unassigned_op_nodes:
                        unassigned_op_node_start_time = property_set["node_start_time"][
                            unassigned_op_node
                        ]
                        unassigned_op_node_end_time = (
                            unassigned_op_node_start_time + unassigned_op_node.op.duration
                        )

                        if unassigned_op_node_start_time > new_delay_end.time:
                            end_event_partition.add(unassigned_op_node)
                            continue

                        if unassigned_op_node_end_time < new_delay_begin.time:
                            begin_event_partition.add(unassigned_op_node)
                            continue

                        # Nodes here overlap narrow section in some way.
                        qarg_idxs = [bit_idx_locs[qarg] for qarg in unassigned_op_node.qargs]
                        begin_qarg_idxs = [
                            bit_idx_locs[qarg]
                            for op_node in begin_event_partition
                            for qarg in op_node.qargs
                        ]
                        end_qarg_idxs = [
                            bit_idx_locs[qarg]
                            for op_node in end_event_partition
                            for qarg in op_node.qargs
                        ]

                        # If adjacent to one side, but not the other, add there
                        if any(
                            cmap.distance(unassigned_qarg, partition_qarg) == 1
                            for unassigned_qarg in qarg_idxs
                            for partition_qarg in begin_qarg_idxs
                        ) and not any(
                            cmap.distance(unassigned_qarg, partition_qarg) == 1
                            for unassigned_qarg in qarg_idxs
                            for partition_qarg in end_qarg_idxs
                        ):
                            begin_event_partition.add(unassigned_op_node)
                            continue

                        if any(
                            cmap.distance(unassigned_qarg, partition_qarg) == 1
                            for unassigned_qarg in qarg_idxs
                            for partition_qarg in end_qarg_idxs
                        ) and not any(
                            cmap.distance(unassigned_qarg, partition_qarg) == 1
                            for unassigned_qarg in qarg_idxs
                            for partition_qarg in begin_qarg_idxs
                        ):
                            end_event_partition.add(unassigned_op_node)
                            continue

                        begin_part_time_overlaps = Counter()
                        end_part_time_overlaps = Counter()

                        # Being greedy here, will depend on the order of exmaining unassigned_op_nodes
                        for begin_part_op_node in begin_event_partition:
                            part_op_node_start_time = property_set["node_start_time"][
                                begin_part_op_node
                            ]
                            part_op_node_end_time = (
                                part_op_node_start_time + begin_part_op_node.op.duration
                            )

                            if unassigned_op_node_start_time > part_op_node_end_time:
                                begin_part_time_overlaps[0] += 1
                            elif unassigned_op_node_end_time < part_op_node_start_time:
                                begin_part_time_overlaps[0] += 1
                            elif unassigned_op_node_start_time < part_op_node_start_time:
                                begin_part_time_overlaps[
                                    unassigned_op_node_end_time - part_op_node_start_time
                                ] += 1
                            else:
                                begin_part_time_overlaps[
                                    part_op_node_end_time - unassigned_op_node_start_time
                                ] += 1

                        for end_part_op_node in end_event_partition:
                            part_op_node_start_time = property_set["node_start_time"][
                                end_part_op_node
                            ]
                            part_op_node_end_time = (
                                part_op_node_start_time + end_part_op_node.op.duration
                            )

                            if unassigned_op_node_start_time > part_op_node_end_time:
                                end_part_time_overlaps[0] += 1
                            elif unassigned_op_node_end_time < part_op_node_start_time:
                                end_part_time_overlaps[0] += 1
                            elif unassigned_op_node_start_time < part_op_node_start_time:
                                end_part_time_overlaps[
                                    unassigned_op_node_end_time - part_op_node_start_time
                                ] += 1
                            else:
                                end_part_time_overlaps[
                                    part_op_node_end_time - unassigned_op_node_start_time
                                ] += 1

                        if (
                            max(begin_part_time_overlaps.keys() | end_part_time_overlaps.keys())
                            < min_joinable_delay_duration
                        ):
                            solo_partitions.append([unassigned_op_node])
                        elif max(begin_part_time_overlaps.keys()) > max(
                            end_part_time_overlaps.keys()
                        ):
                            begin_event_partition.add(unassigned_op_node)
                        else:
                            end_event_partition.add(unassigned_op_node)

                    for partition in [begin_event_partition, end_event_partition] + solo_partitions:
                        part_sorted_delay_events = sorted(
                            (
                                DelayEvent(
                                    event_type,
                                    (
                                        property_set["node_start_time"][op_node]
                                        if event_type == EventType.BEGIN
                                        else property_set["node_start_time"][op_node]
                                        + op_node.op.duration
                                    ),
                                    op_node,
                                )
                                for op_node in partition
                                for event_type in (EventType.BEGIN, EventType.END)
                            ),
                            key=DelayEvent.sort_key,
                        )
                        part_adjacent_delay_blocks = _collect_adjacent_delay_blocks(
                            part_sorted_delay_events, cmap, bit_idx_locs
                        )
                        part_grouped_delay_blocks = _group_delay_blocks(part_adjacent_delay_blocks)

                        grouped_delay_blocks.extend(part_grouped_delay_blocks)

    return closed_joint_delays


def _ungroup_adj_grouped_delay_block(grouped_delay_block):
    rtn = AdjacentDelayBlock([], set())
    for event in grouped_delay_block.events:
        for op_node in event.op_nodes:
            rtn.events.append(DelayEvent(type=event.type, time=event.time, op_node=op_node))
    return rtn


def _find_widest_connected_joint_delay(grouped_delay):

    # Find widest point, incr num_qubits on open, decr on close
    current_open_qubits = set()
    current_open_op_nodes = []

    widest = CollectedJointDelay(
        num_qubits=len(current_open_qubits),
        begin_delay_event=None,
        end_delay_event=None,
        open_op_nodes=current_open_op_nodes.copy(),
    )

    for grouped_delay_event in grouped_delay.events:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Considering event: %s with widest at entry: %s",
                pformat(grouped_delay_event),
                widest,
            )

        event_qubits = grouped_delay_event.qargs

        if grouped_delay_event.type == EventType.BEGIN:
            if not current_open_qubits.isdisjoint(event_qubits):
                raise RuntimeError("re-opening open qubits?")

            current_open_qubits.update(event_qubits)
            current_open_op_nodes.extend(grouped_delay_event.op_nodes)

            # KDK Need to check connectedness too
            if len(current_open_qubits) > widest.num_qubits:
                widest = CollectedJointDelay(
                    num_qubits=len(current_open_qubits),
                    begin_delay_event=grouped_delay_event,
                    end_delay_event=None,
                    open_op_nodes=current_open_op_nodes.copy(),
                )

        else:  # grouped_delay_event.type == EventType.END
            if not event_qubits.issubset(current_open_qubits):
                raise RuntimeError(f"Closing qubits that are not open: {event_qubits}")

            if widest.end_delay_event is None:
                widest.end_delay_event = grouped_delay_event

            current_open_qubits -= event_qubits
            for op_node in grouped_delay_event.op_nodes:
                current_open_op_nodes.remove(op_node)

    return widest
