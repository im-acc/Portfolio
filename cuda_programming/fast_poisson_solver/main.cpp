#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "solvers.h"
#define N 1500

int main()
{
    float* sol = (float*)calloc((N + 2) * (N + 2), sizeof(float));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    host_solver(sol, N, 2000);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //printSol(sol,N);
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    return 0;
}