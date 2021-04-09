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

"""
Fake Casablanca device (7 qubit).
"""

import os
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend


class FakeCasablanca(FakePulseBackend):
    """A fake 7 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_casablanca.json"
    props_filename = "props_casablanca.json"
    defs_filename = "defs_casablanca.json"
    backend_name = "fake_casablanca"
