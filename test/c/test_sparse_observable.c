#include <stdio.h>
#include <complex.h>
#include "qiskit.h"

int test_zero()
{
    SparseObservable *obs = obs_zero(100);
    obs_print(obs);
    obs_deallocate(obs);
    return 0;
}

int test_identity()
{
    SparseObservable *obs = obs_identity(100);
    obs_print(obs);
    obs_deallocate(obs);
    return 0;
}

int test_copy()
{
    SparseObservable *obs = obs_identity(100);
    SparseObservable *copied = obs_copy(obs);
    obs_print(copied);
    obs_deallocate(obs);
    obs_deallocate(copied);
    return 0;
}

int test_add()
{
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
    uint64_t num_terms;

    SparseObservable *zero = obs_zero(100);
    num_terms = obs_num_terms(zero);
    obs_deallocate(zero);

    SparseObservable *iden = obs_identity(100);
    num_terms = obs_num_terms(iden);
    obs_deallocate(iden);

    return 0;
}

int test_num_qubits()
{
    uint32_t num_qubits;

    SparseObservable *obs = obs_zero(1);
    num_qubits = obs_num_qubits(obs);
    obs_deallocate(obs);

    SparseObservable *obs100 = obs_zero(100);
    num_qubits = obs_num_qubits(obs100);
    obs_deallocate(obs100);

    return 0;
}

int run(char* name, int result)
{
    int did_fail = 0;
    char* msg = "OK";
    if (result > 0)
    {
        msg = "FAILED with unknown error";
    }
    fprintf(stderr, "--- %-30s: %s\n", name, msg);
    fflush(stderr);

    return did_fail;
}

int test_sparse_observable()
{
    int num_failed = 0;

    num_failed += run("test_zero", test_zero());
    num_failed += run("test_identity", test_identity());
    num_failed += run("test_add", test_add());
    num_failed += run("test_mult_real", test_mult_real());
    num_failed += run("test_mult_complex", test_mult_complex());
    num_failed += run("test_canonicalize", test_canonicalize());
    num_failed += run("test_copy", test_copy());
    num_failed += run("test_num_terms", test_num_terms());
    num_failed += run("test_num_qubits", test_num_qubits());

    fprintf(stderr, "=== Number of failed subtests: %i", num_failed);

    return num_failed;
}
