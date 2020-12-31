
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <stdlib.h>
#include <stdio.h>
#define TX 32 // threads per block in x
#define TY 32 // threads per block in y


// non-homogenous terme
__device__ float d_f(float x, float y) {
	return sinpif(x) * sinpif(y);
}


__global__ void initSolKernel(float* d_v, int w, int h) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = w * r + c;

	if ((c >= w) || (r >= h)) return;
	d_v[i] = 0;
}

__global__ void JacobiKernel(float* d_v, int w, int h, float h1, float h2) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = w * r + c;
	if (( r<= 0 || c >= w-1) || ( r<= 1 || r >= h-1)) return;
	d_v[i] = (d_v[i - w] + d_v[i - 1] + d_v[i + w] + d_v[i + 1] - h2 * d_f(h1*c, h1*r)) * 0.25;
	//printf("i = %2d: dist from %f to %f is %f.\n", i, d_v[i], 2, 2);
}

// Solver laplace(v) = f(x,y)
// [0,1]x[0,1] Square boundary valued at 0 ( v(B) = 0 ) 
void device_solver(float* v, int n, int niter) {

	int W = n + 2;
	int H = n + 2;

	float* d_v = 0;
	cudaMallocManaged(&d_v, W * H * sizeof(float));
	
	int bx = (W + TX - 1) / TX;
	int by = (H + TY - 1) / TY;

	dim3 gridSize(bx, by);
	dim3 blockSize(TX, TY);

	// init solution to 0
	initSolKernel <<< gridSize, blockSize >> > (d_v, W, H);
	cudaDeviceSynchronize();
	float h1 = 1.0/( (float) n);
	float h2 = h1 * h1;
	
	for (int k = 0; k < niter; k++)
	{
		JacobiKernel <<< gridSize, blockSize >> > (d_v, W, H, h1, h2);
	}
	cudaDeviceSynchronize();
	
	cudaMemcpy(v, d_v, W * H * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_v);
}
