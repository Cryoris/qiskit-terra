#include <stdio.h>
#include <complex.h>
#include "common.h"
#include "qiskit.h"

int test_zero()
{
    SparseObservable *obs = obs_zero(100);
    uint64_t num_terms = obs_num_terms(obs);
    uint32_t num_qubits = obs_num_qubits(obs);
    obs_deallocate(obs);

    if (num_terms != 0 || num_qubits != 100)
    {
        return EqualityError;
    }
    return 0;
}

int test_identity()
{
    SparseObservable *obs = obs_identity(100);
    uint64_t num_terms = obs_num_terms(obs);
    uint32_t num_qubits = obs_num_qubits(obs);
    obs_deallocate(obs);

    if (num_terms != 1 || num_qubits != 100)
    {
        return EqualityError;
    }
    return 0;
}

int test_copy()
{
    SparseObservable *obs = obs_identity(100);
    SparseObservable *copied = obs_copy(obs);

    bool are_equal = obs_equal(obs, copied);

    obs_deallocate(obs);
    obs_deallocate(copied);

    if (!are_equal)
    {
        return EqualityError;
    }

    return 0;
}

int test_add()
{
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    uint64_t num_terms = obs_num_terms(obs);

    obs_deallocate(left);
    obs_deallocate(right);
    obs_deallocate(obs);

    if (num_terms != 2)
    {
        return EqualityError;
    }

    return 0;
}

int test_mult_real()
{
    SparseObservable *obs = obs_identity(100);

    double coeff = 2;
    SparseObservable *result = obs_multiply(obs, coeff);

    obs_print(result);
    // TODO: actually perform some equality check (requires obs_term)

    obs_deallocate(obs);
    obs_deallocate(result);

    return 0;
}

int test_mult_complex()
{
    SparseObservable *obs = obs_identity(100);

    complex double coeff = 2 + 2 * I;
    SparseObservable *result = obs_multiply(obs, coeff);

    obs_print(result);
    // TODO: actually perform some equality check (requires obs_term)

    obs_deallocate(obs);
    obs_deallocate(result);

    return 0;
}

int test_canonicalize()
{
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    double tol = 1e-5;
    SparseObservable *simplified = obs_canonicalize(obs, tol);

    obs_print(simplified);
    // TODO: actually perform some equality check (requires obs_term)

    obs_deallocate(obs);
    obs_deallocate(right);
    obs_deallocate(left);
    obs_deallocate(simplified);

    return 0;
}

int test_num_terms()
{
    int result = Ok;
    uint64_t num_terms;

    SparseObservable *zero = obs_zero(100);
    num_terms = obs_num_terms(zero);
    if (num_terms != 0)
    {
        result = EqualityError;
    }
    obs_deallocate(zero);

    SparseObservable *iden = obs_identity(100);
    num_terms = obs_num_terms(iden);
    if (num_terms != 1)
    {
        result = EqualityError;
    }
    obs_deallocate(iden);

    return result;
}

int test_num_qubits()
{
    int result = Ok;
    uint32_t num_qubits;

    SparseObservable *obs = obs_zero(1);
    num_qubits = obs_num_qubits(obs);
    if (num_qubits != 1)
    {
        result = EqualityError;
    }
    obs_deallocate(obs);

    SparseObservable *obs100 = obs_zero(100);
    num_qubits = obs_num_qubits(obs100);
    if (num_qubits != 100)
    {
        result = EqualityError;
    }
    obs_deallocate(obs100);

    return result;
}

int test_custom_build()
{
    u_int32_t num_qubits = 100;
    SparseObservable *obs = obs_zero(num_qubits);

    complex double coeff = 1;

    BitTermVec *bits = bit_terms_new(); // could use with_capacity here too, but we test new()
    bit_terms_push(bits, BitTerm_X);    // these enums are defined in BitTerm
    bit_terms_push(bits, BitTerm_Y);
    bit_terms_push(bits, BitTerm_Z);

    IndexVec *indices = indices_with_capacity(3);
    indices_push(indices, 0);
    indices_push(indices, 1);
    indices_push(indices, 2);

    obs_push_copy(obs, bits, indices, coeff);
    obs_push_consume(obs, bits, indices, coeff); // consumes the bits and indices vectors

    obs_print(obs); // TODO do some check

    double tol = 1e-6;
    SparseObservable *simplified = obs_canonicalize(obs, tol);
    obs_print(simplified); // TODO do some check

    obs_deallocate(obs);

    return 0;
}

int test_term()
{
    SparseObservable *obs = obs_identity(100);

    BitTermVec *bits = bit_terms_with_capacity(3);
    bit_terms_push(bits, BitTerm_X);
    bit_terms_push(bits, BitTerm_Y);
    bit_terms_push(bits, BitTerm_Z);

    IndexVec *indices = indices_with_capacity(3);
    indices_push(indices, 0);
    indices_push(indices, 1);
    indices_push(indices, 2);

    complex double coeff = 1 + I;

    obs_push_consume(obs, bits, indices, coeff);
    obs_print(obs);

    uint64_t num_terms = obs_num_terms(obs);
    for (uint64_t i = 0; i < num_terms; i++)
    {
        SparseTerm *term = obs_term(obs, i);
        obsterm_print(term);
        uint32_t nni = obsterm_nni(term);
        printf("nni: %u\n", nni); // todo do some check

        for (uint32_t n = 0; n < nni; n++)
        {
            PauliTerm *pterm = obsterm_pauli(term, n);
            printf("Pauli: %i Index: %i\n", pterm->bit_term, pterm->index); // todo some check
        }

        obsterm_deallocate(term);
    }

    obs_deallocate(obs);

    return 0;
}

int test_sparse_observable()
{
    int num_failed = 0;
    num_failed += RUN_TEST(test_zero);
    num_failed += RUN_TEST(test_identity);
    num_failed += RUN_TEST(test_add);
    num_failed += RUN_TEST(test_mult_real);
    num_failed += RUN_TEST(test_mult_complex);
    num_failed += RUN_TEST(test_canonicalize);
    num_failed += RUN_TEST(test_copy);
    num_failed += RUN_TEST(test_num_terms);
    num_failed += RUN_TEST(test_num_qubits);
    num_failed += RUN_TEST(test_custom_build);
    num_failed += RUN_TEST(test_term);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);

    return num_failed;
}
