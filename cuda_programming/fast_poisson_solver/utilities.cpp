#include <iostream>
#include <stdlib.h>

void printSol(float* sol, int n) {
    int d = n + 2;
    std::cout << "[";
    for (int i = 0; i < (n + 2) * (n + 2); i += d)
    {
        std::cout << "[";
        for (int j = 0; j < n + 2; j++)
        {
            std::cout << sol[i + j];
            if (j != n + 1)
                std::cout << ",";
        }
        std::cout << "]";
        if (i != (n + 2) * (n + 1))
            std::cout << ",";
    }
    std::cout << "]";
}