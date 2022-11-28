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
===========================================
Circuit Synthesis (:mod:`qiskit.synthesis`)
===========================================

.. currentmodule:: qiskit.synthesis

Evolution Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
   ProductFormula
   LieTrotter
   SuzukiTrotter
   MatrixExponential

Permutation Synthesis
=====================

.. autosummary::
   :toctree: ../stubs/

   synth_permutation_depth_lnn_kms

"""

from .evolution import (
    EvolutionSynthesis,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    MatrixExponential,
    QDrift,
)

from .permutation import synth_permutation_depth_lnn_kms
