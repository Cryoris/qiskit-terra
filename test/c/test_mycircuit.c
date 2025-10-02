// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <qiskit.h>

static int test_simple_case(void) {
    QkCircuit *qc = qk_circuit_library_mycircuit();
    size_t num_instructions = qk_circuit_num_instructions(qc);

    int result = Ok;
    if (num_instructions != 3) {
        result = EqualityError;
    }

    qk_circuit_free(qc);
    return result;
}

int test_mycircuit(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_simple_case);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
