# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object

from time import time
import numpy as np
from utility_scale import UtilityScaleBenchmarks

reps = 10

basis_gate = "cx"
bench = UtilityScaleBenchmarks()
bench.setup(basis_gate)


def timeit(label, fun):
    times = []
    for _ in range(reps):
        start = time()
        fun(basis_gate)
        times.append(time() - start)

    print(f"{label}: {np.mean(times)} +- {np.std(times)}")
    return np.mean(times)


print("-- Times:")
total = 0
total += timeit("QFT", bench.time_qft)
total += timeit("Heisen", bench.time_square_heisenberg)
total += timeit("QAOA", bench.time_qaoa)
total += timeit("QV", bench.time_qv)
total += timeit("SU2", bench.time_circSU2)
total += timeit("BV", bench.time_bv_100)
total += timeit("BV-like", bench.time_bvlike)
print("Total:", total)
print()

print("-- Depths:")
print("QFT:", bench.track_qft_depth(basis_gate))
print("Heisen:", bench.track_square_heisenberg_depth(basis_gate))
print("QAOA:", bench.track_qaoa_depth(basis_gate))
print("QV:", bench.track_qv_depth(basis_gate))
print("SU2:", bench.track_circSU2_depth(basis_gate))
print("BV:", bench.track_bv_100_depth(basis_gate))
print("BV-like:", bench.track_bvlike_depth(basis_gate))
