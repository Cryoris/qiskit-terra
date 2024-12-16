#include <stdio.h>
#include <complex.h>
#include "qiskit.h"

int test_zero()
{
    printf("\n-- test_zero\n");
    SparseObservable *obs = obs_zero(100);
    obs_print(obs);
    obs_deallocate(obs);
    return 0;
}

int test_identity()
{
    printf("\n-- test_identity\n");
    SparseObservable *obs = obs_identity(100);
    obs_print(obs);
    obs_deallocate(obs);
    return 0;
}

int test_copy()
{
    printf("\n-- test_copy\n");
    SparseObservable *obs = obs_identity(100);
    SparseObservable *copied = obs_copy(obs);
    obs_print(copied);
    obs_deallocate(obs);
    obs_deallocate(copied);
    return 0;
}

int test_add()
{
    printf("\n-- test_add\n");
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    obs_print(obs);

    obs_deallocate(left);
    obs_deallocate(right);
    obs_deallocate(obs);

    return 0;
}

int test_mult_real()
{
    printf("\n-- test_mult_real\n");
    SparseObservable *obs = obs_identity(100);

    double coeff = 2;
    SparseObservable *result = obs_multiply(obs, coeff);

    obs_print(result);

    obs_deallocate(obs);
    obs_deallocate(result);

    return 0;
}

int test_mult_complex()
{
    printf("\n-- test_mult_complex\n");
    SparseObservable *obs = obs_identity(100);

    complex double coeff = 2 + 2 * I;
    SparseObservable *result = obs_multiply(obs, coeff);

    obs_print(result);

    obs_deallocate(obs);
    obs_deallocate(result);

    return 0;
}

int test_canonicalize()
{
    printf("\n-- test_canonicalize\n");
    SparseObservable *left = obs_identity(100);
    SparseObservable *right = obs_identity(100);
    SparseObservable *obs = obs_add(left, right);

    double tol = 1e-5;
    SparseObservable *simplified = obs_canonicalize(obs, tol);

    obs_print(simplified);

    obs_deallocate(obs);
    obs_deallocate(right);
    obs_deallocate(left);
    obs_deallocate(simplified);

    return 0;
}

int test_num_terms()
{
    printf("\n-- test_num_terms\n");

    uint64_t num_terms;

    SparseObservable *zero = obs_zero(100);
    num_terms = obs_num_terms(zero);
    printf("zero: %llu\n", num_terms);
    obs_deallocate(zero);

    SparseObservable *iden = obs_identity(100);
    num_terms = obs_num_terms(iden);
    printf("identity: %llu\n", num_terms);
    obs_deallocate(iden);

    return 0;
}

int test_num_qubits()
{
    printf("\n-- test_num_qubits\n");

    uint32_t num_qubits;

    SparseObservable *obs = obs_zero(1);
    num_qubits = obs_num_qubits(obs);
    printf("1 qubit: %u\n", num_qubits);
    obs_deallocate(obs);

    SparseObservable *obs100 = obs_zero(100);
    num_qubits = obs_num_qubits(obs100);
    printf("100 qubits: %u\n", num_qubits);
    obs_deallocate(obs100);

    return 0;
}

int main()
{
    // BitTerm *bit = bit_term(1);
    // printf("%i", *bit);
    // bit_term_print(bit);
    // bit_term_deallocate(bit);

    test_zero();
    test_identity();
    test_add();
    test_mult_real();
    test_mult_complex();
    test_canonicalize();
    test_copy();
    test_num_terms();
    test_num_qubits();

    return 0;
}