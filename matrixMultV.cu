#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

#define X 4 // size for N x N matrices

using namespace std;
		
// Matrix multiplication kernel - thread specification
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	// product stores the Pd element that is computed by the thread
	float product = 0;

	// 2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	for (int i = 0; i < Width; ++i)
	{
		float MdElement = Md[ty * Width + i]; // incrementing horizontally on a 2d rep.
		float NdElement = Nd[i * Width + tx]; // incrementing downwards on a 2d rep.
		product += MdElement * NdElement;
	}

	// Write the matrix to device memory each thread writes one element
	Pd[ty * Width + tx] = product;
}

void MatrixMultiplication(float* M, float* N, float* P, int Width) 
{
	int size = Width * Width * sizeof(float);
	float *Md, *Nd, *Pd;
	
	// Allocate device memory for M, N, and P
	cudaMalloc(&Md, size);
	cudaMalloc(&Nd, size);
	cudaMalloc(&Pd, size);

	// copy M and N to allocated device memory locations
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Pd, P, size, cudaMemcpyHostToDevice);

	// Kernel invocation code - to have the device to perform the actual matrix multiplication
	// Setup the execution configuration
	dim3 dimBlock(Width, Width);
	dim3 dimGrid(1,1);

	// Launch the device computation threads
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md,Nd,Pd,Width);

	// copy P from the device memory
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	
	// Free device matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd); 
}

void initialize(float* arr, int size) 
{
	for (int i = 0; i < size; i++) 
	{
		arr[i] = 1.0f;
	}
}

void validate(float *a, float *b, float *c, int n){
    	// serial implementation
    	for (size_t i = 0; i < n; i++){ // rows
        	for (size_t j = 0; j < n; j++){ // columns
            		int temp = 0;
            		for (size_t k = 0; k < n; k++){
                		temp += a[i*n+k] * b[k*n+j];
            		}
            		// Check against the CPU result
            		assert(temp == c[i * n + j]);
        		}
    	}
	printf("Answer Validated\n");
}

int main(void) 
{
	int matrixSize = X * X;
	int Width = X;

	// Allocate and initialize the matrices M, N, P
	float* Mh = (float*)malloc(matrixSize*sizeof(float));
	float* Nh = (float*)malloc(matrixSize*sizeof(float));
	float* Ph = (float*)malloc(matrixSize*sizeof(float));

	// I/O to read the input matrices M and N
	initialize(Mh, matrixSize);
	initialize(Nh, matrixSize);
	std::cout << "test" << std::endl;
	MatrixMultiplication(Mh, Nh, Ph, Width);

	// I/O to write the output matrix P
    	puts("CPU: Validating...");
    	int counter = 0;
	for (int i = 0; i < matrixSize; i++)
	{
		if (counter == X)  
		{
			counter = 0;
			printf("\n");
		}
		printf("%.2f ", Ph[i]);
		counter++;
	}
	printf("\n");
    
    	validate(Mh, Nh, Ph, X);
	delete[] Mh; delete[] Nh; delete[] Ph;
	return 0;
}
