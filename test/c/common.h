#include<stdio.h>

enum TestResult {
    Ok,
    EqualityError,
};

int run(char* name, int result)
{
    int did_fail = 1;
    char* msg;
    if (result == Ok)
    {
        did_fail = 0;
        msg = "Ok";
    } else if (result == EqualityError)
    {
        msg = "FAILED with an EqualityError";
    } else {
        msg = "FAILED with unknown error";
    }
    fprintf(stderr, "--- %-30s: %s\n", name, msg);
    fflush(stderr);

    return did_fail;
}
