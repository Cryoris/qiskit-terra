// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include "qiskit.h"
#include <complex.h>
#include <stdio.h>

int test_zero() {
    SparseObservable *obs = obs_zero(100);
    uint64_t num_terms = obs_num_terms(obs);
    uint32_t num_qubits = obs_num_qubits(obs);
    obs_free(obs);

    if (num_terms != 0 || num_qubits != 100) {
        return EqualityError;
    }
    return 0;
}

int test_identity() {
    SparseObservable *obs = obs_identity(100);
    uint64_t num_terms = obs_num_terms(obs);
    uint32_t num_qubits = obs_num_qubits(obs);
    obs_free(obs);

    if (num_terms != 1 || num_qubits != 100) {
        return EqualityError;
    }
    return 0;
}

int test_copy() {
    SparseObservable *obs = obs_identity(100);
    SparseObservable *copied = obs_copy(obs);

    bool are_equal = obs_equal(obs, copied);

    obs_free(obs);
    obs_free(copied);

    if (!are_equal) {
        return EqualityError;
    }

    return 0;
}

int test_add() {
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    uint64_t num_terms = obs_num_terms(obs);

    obs_free(left);
    obs_free(right);
    obs_free(obs);

    if (num_terms != 2) {
        return EqualityError;
    }

    return 0;
}

int test_mult() {
    complex double coeffs[3] = {2, 2 * I, 2 + 2 * I};

    for (int i = 0; i < 3; i++) {
        SparseObservable *obs = obs_identity(100);

        SparseObservable *result = obs_multiply(obs, coeffs[i]);

        // construct the expected observable: coeff * Id
        SparseObservable *expected = obs_zero(100);
        PauliTermVec *paulis = paulis_new();
        obs_push_consume(expected, paulis, coeffs[i]);

        // perform the check
        bool is_equal = obs_equal(expected, result);

        // deallocate before returning
        obs_free(obs);
        obs_free(result);
        obs_free(expected);

        if (!is_equal) {
            return EqualityError;
        }
    }

    return 0;
}

int test_canonicalize() {
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    double tol = 1e-5;
    SparseObservable *simplified = obs_canonicalize(obs, tol);

    // construct the expected observable: 2 * Id
    SparseObservable *expected = obs_zero(100);
    PauliTermVec *paulis = paulis_new();
    obs_push_consume(expected, paulis, 2.0);

    bool is_equal = obs_equal(expected, simplified);

    obs_free(obs);
    obs_free(right);
    obs_free(left);
    obs_free(simplified);
    obs_free(expected);

    if (!is_equal) {
        return EqualityError;
    }

    return 0;
}

int test_num_terms() {
    int result = Ok;
    uint64_t num_terms;

    SparseObservable *zero = obs_zero(100);
    num_terms = obs_num_terms(zero);
    if (num_terms != 0) {
        result = EqualityError;
    }
    obs_free(zero);

    SparseObservable *iden = obs_identity(100);
    num_terms = obs_num_terms(iden);
    if (num_terms != 1) {
        result = EqualityError;
    }
    obs_free(iden);

    return result;
}

int test_num_qubits() {
    int result = Ok;
    uint32_t num_qubits;

    SparseObservable *obs = obs_zero(1);
    num_qubits = obs_num_qubits(obs);
    if (num_qubits != 1) {
        result = EqualityError;
    }
    obs_free(obs);

    SparseObservable *obs100 = obs_zero(100);
    num_qubits = obs_num_qubits(obs100);
    if (num_qubits != 100) {
        result = EqualityError;
    }
    obs_free(obs100);

    return result;
}

int test_custom_build() {
    u_int32_t num_qubits = 100;
    SparseObservable *obs = obs_zero(num_qubits);

    complex double coeff = 1;

    PauliTermVec *paulis = paulis_new(); // could use with_capacity here, but we test new()
    paulis_push(paulis, BitTerm_X, 0);
    paulis_push(paulis, BitTerm_Y, 1);
    paulis_push(paulis, BitTerm_Z, 2);

    obs_push_copy(obs, paulis, coeff);
    obs_push_consume(obs, paulis, coeff); // consumes the bits and indices vectors

    double tol = 1e-6;
    SparseObservable *simplified = obs_canonicalize(obs, tol);

    uint64_t num_terms = obs_num_terms(obs);
    uint64_t num_terms_simplified = obs_num_terms(simplified);

    obs_free(obs);
    obs_free(simplified);

    if (num_terms != 2 || num_terms_simplified != 1) {
        return EqualityError;
    }

    return 0;
}

int test_term() {
    SparseObservable *obs = obs_identity(100);

    PauliTermVec *paulis = paulis_with_capacity(3);
    paulis_push(paulis, BitTerm_X, 0);
    paulis_push(paulis, BitTerm_Y, 1);
    paulis_push(paulis, BitTerm_Z, 2);

    complex double coeff = 1 + I;

    obs_push_consume(obs, paulis, coeff);

    // some placeholders to store the results
    int nnis[2] = {-1, -1};
    int bits[3] = {-1, -1, -1};
    int indices[3] = {-1, -1, -1};

    uint64_t num_terms = obs_num_terms(obs);
    for (uint64_t i = 0; i < num_terms; i++) {
        SparseTerm *term = obs_term(obs, i);
        uint32_t nni = obsterm_nni(term);
        nnis[i] = nni; // store to compare later

        for (uint32_t n = 0; n < nni; n++) {
            PauliTerm *pterm = obsterm_pauli(term, n);

            // this loop is only called once, so we can use ``n`` to index here
            bits[n] = pterm->bit_term;
            indices[n] = pterm->index;

            pauli_free(pterm);
        }

        obsterm_free(term);
    }

    obs_free(obs);

    int result = 0;
    int expected_nnis[2] = {0, 3};
    int expected_bits[3] = {BitTerm_X, BitTerm_Y, BitTerm_Z};
    int expected_indices[3] = {0, 1, 2};

    // check number of terms
    if (num_terms != 2) {
        result = EqualityError;
    }

    // check NNIs
    for (int i = 0; i < 2; i++) {
        if (nnis[i] != expected_nnis[i]) {
            result = EqualityError;
        }
    }

    // check bit terms and indices
    for (int n = 0; n < 3; n++) {
        if (indices[n] != expected_indices[n] || bits[n] != expected_bits[n]) {
            result = EqualityError;
        }
    }

    return result;
}

int test_sparse_observable() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_zero);
    num_failed += RUN_TEST(test_identity);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_mult);
    num_failed += RUN_TEST(test_canonicalize);
    num_failed += RUN_TEST(test_copy);
    num_failed += RUN_TEST(test_num_terms);
    num_failed += RUN_TEST(test_num_qubits);
    num_failed += RUN_TEST(test_custom_build);
    num_failed += RUN_TEST(test_term);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
