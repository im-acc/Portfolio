#pragma once

void host_solver(float* v, int n, int niter);
void device_solver(float* v, int n, int niter);

// utilities
void printSol(float* sol, int n);