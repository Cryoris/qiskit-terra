#include "qiskit.h"

int main()
{
    // SparseObservable *obs = obs_zero(100);
    // obs_print(obs);
    // obs_deallocate(obs);
    BitTerm *bit = bit_term(1);
    printf("%i", *bit);
    bit_term_print(bit);
    bit_term_deallocate(bit);
    // enum BitTerm bit;
    // bit = Y;

    return 0;
}