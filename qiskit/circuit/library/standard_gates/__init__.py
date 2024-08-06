# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Standard gates
"""

from .h import HGate, CHGate
from .i import IGate
from .p import PhaseGate, CPhaseGate, MCPhaseGate
from .r import RGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .rzx import RZXGate
from .xx_minus_yy import XXMinusYYGate
from .xx_plus_yy import XXPlusYYGate
from .ecr import ECRGate
from .s import SGate, SdgGate, CSGate, CSdgGate
from .swap import SwapGate, CSwapGate
from .iswap import iSwapGate
from .sx import SXGate, SXdgGate, CSXGate
from .dcx import DCXGate
from .t import TGate, TdgGate
from .u import UGate, CUGate
from .u1 import U1Gate, CU1Gate, MCU1Gate
from .u2 import U2Gate
from .u3 import U3Gate, CU3Gate
from .x import XGate, CXGate, CCXGate, C3XGate, C3SXGate, C4XGate, RCCXGate, RC3XGate
from .x import MCXGate, MCXGrayCode, MCXRecursive, MCXVChain
from .y import YGate, CYGate
from .z import ZGate, CZGate, CCZGate
from .global_phase import GlobalPhaseGate
from .multi_control_rotation_gates import mcrx, mcry, mcrz


def get_standard_gate_name_mapping():
    """Return a dictionary mapping the name of standard gates and instructions to an object for
    that name."""
    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.measure import Measure
    from qiskit.circuit.delay import Delay
    from qiskit.circuit.reset import Reset

    # Standard gates library mapping, multicontrolled gates not included since they're
    # variable width
    gates = [
        C3SXGate(),
        C3XGate(),
        C4XGate(),
        CCXGate(),
        CCZGate(),
        CHGate(),
        CPhaseGate(Parameter("ϴ")),
        CRXGate(Parameter("ϴ")),
        CRYGate(Parameter("ϴ")),
        CRZGate(Parameter("ϴ")),
        CSdgGate(),
        CSGate(),
        CSwapGate(),
        CSXGate(),
        CU1Gate(Parameter("λ")),
        CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
        CXGate(),
        CYGate(),
        CZGate(),
        DCXGate(),
        Delay(Parameter("t")),
        ECRGate(),
        GlobalPhaseGate(Parameter("ϴ")),
        HGate(),
        IGate(),
        iSwapGate(),
        Measure(),
        PhaseGate(Parameter("ϴ")),
        RC3XGate(),
        RCCXGate(),
        Reset(),
        RGate(Parameter("ϴ"), Parameter("φ")),
        RXGate(Parameter("ϴ")),
        RXXGate(Parameter("ϴ")),
        RYGate(Parameter("ϴ")),
        RYYGate(Parameter("ϴ")),
        RZGate(Parameter("λ")),
        RZXGate(Parameter("ϴ")),
        RZZGate(Parameter("ϴ")),
        SdgGate(),
        SGate(),
        SwapGate(),
        SXdgGate(),
        SXGate(),
        TdgGate(),
        TGate(),
        U1Gate(Parameter("λ")),
        U2Gate(Parameter("φ"), Parameter("λ")),
        U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        XGate(),
        XXMinusYYGate(Parameter("ϴ"), Parameter("β")),
        XXPlusYYGate(Parameter("ϴ"), Parameter("β")),
        YGate(),
        ZGate(),
    ]
    name_mapping = {gate.name: gate for gate in gates}
    return name_mapping
