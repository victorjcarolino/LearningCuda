#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

#define N 1024 // size for N x N matrices

__global__ void MatrixMultiplication(float* M, float* N, float* P, int Width) 
{
	int size = Width * Width * sizeof(float);
	float* Md, Nd, Pd;
	
	// 2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// product stores the Pd element that is computed by the thread
	float product = 0;

	// Allocate device memory for M, N, and P
	cudaMalloc((void**)Md, size);
	cudaMalloc((void**)Nd, size);
	cudaMalloc((void**)Pd, size);

	// copy M and N to allocated device memory locations
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Pd, P, size, cudaMemcpyHostToDevice);

	// Kernel invocation code - to have the device to perform the actual matrix multiplication
	for (int i = 0; i < Width; ++i)
	{
		float MdElement = Md[ty * Width + i]; // incrementing horizontally on a 2d rep.
		float NdElement = Nd[i * Width + tx]; // incrementing downwards on a 2d rep.
		product += MdElement * NdElement;
	}

	// Write the matrix to device memory each thread writes one element
	Pd[ty * Width + tx] = product;

	// copy P from the device memory
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	
	// Free device matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd); 
}

float* initialize(float* arr, int size) 
{
	for (int i = 0; i < size; i++) 
	{
		arr[i] = 1.0f;
	}
	return arr;
}

void validate()
{
}

int main(void) 
{
	int matrixSize = N * N;

	// Allocate and initialize the matrices M, N, P
	float Mh[N * N * sizeof(float);
	float Nh[N * N * sizeof(float);
	float Ph[N * N * sizeof(float);

	Mh = initialize(Mh, matrixSize);
	Nh = initialize(Nh, matrixSize);
	Ph = initialize(Ph, matrixSize);

	// I/O to read the input matrices M and N

	// M * N on the device
	
	// I/O to write the output matrix P
	// Free matrices M, N, P

	return 0;
}
