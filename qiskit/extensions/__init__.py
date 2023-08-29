# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================================
Quantum Circuit Extensions (:mod:`qiskit.extensions`)
=====================================================

.. currentmodule:: qiskit.extensions

Unitary Extensions
==================

.. autosummary::
   :toctree: ../stubs/

   UnitaryGate
   HamiltonianGate
   SingleQubitUnitary

Simulator Extensions
====================

.. autosummary::
   :toctree: ../stubs/

   Snapshot

Initialization
==============

.. autosummary::
   :toctree: ../stubs/

   Initialize

Uniformly Controlled Rotations
==============================

.. autosummary::
   :toctree: ../stubs

   UCPauliRotGate
   UCRXGate
   UCRYGate
   UCRZGate

Exceptions
==========

The additional gates in this module will tend to raise a custom exception when they encounter
problems.

.. autoexception:: ExtensionError
"""

# pylint: disable=wrong-import-position

import warnings

# put the module level warning first
warnings.warn(
    "The qiskit.extensions module is deprecated since Qiskit 0.45.0. It will be removed no sooner "
    "than 3 months after the release date.",
    stacklevel=2,
    category=DeprecationWarning,
)


# import all standard gates
from qiskit.circuit.library.standard_gates import *
from qiskit.circuit.barrier import Barrier

from .exceptions import ExtensionError
from .quantum_initializer import (
    Initialize,
    SingleQubitUnitary,
    UCPauliRotGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
)
from .unitary import UnitaryGate
from .hamiltonian_gate import HamiltonianGate
from .simulator import Snapshot


def _deprecate_extension(what, has_replacement):
    msg = (
        f"The qiskit.extensions.{what} object is deprecated since Qiskit 0.45.0. It will be "
        "removed no sooner than 3 months after the release date."
    )
    if has_replacement:
        msg += f" Instead, use qiskit.extensions.{what} as replacement."

    warnings.warn(msg, stacklevel=3, category=DeprecationWarning)
