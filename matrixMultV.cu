#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

__device__ void MatrixMultiplication(float* M, float* N, float* P, int Width) {
	int size = Width * Width * sizeof(float);
	float* Md, Nd, Pd;
	
	// Allocate device memory for M, N, and P
	// copy M and N to allocated device memory locations

	// Kernel invocation code - to have the device to perform the actual matrix multiplication

	// copy P from the device memory
	// Free device matrices
}

int main(void) {
	// Allocate and initialize the matrices M, N, P
	// I/O to read the input matrices M and N

	// M * N on the device
	
	// I/O to write the output matrix P
	// Free matrices M, N, P

	return 0;
}
