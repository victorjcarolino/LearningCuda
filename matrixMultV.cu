#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

#define X 16 // size for N x N matrices
#define TILE_WIDTH 2 // size for the tiles

using namespace std;
		
// Matrix multiplication kernel - thread specification
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // shared array of Md elements to lessen global mem accesses
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // shared array of Nd elements to lessen global mem accesses
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float product = 0;

	// Loop over the Md and Nd tiles requried to compute the Pd element
	for (int i = 0; i < Width/TILE_WIDTH; ++i) 
	{
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[row * Width + (i * TILE_WIDTH + ty)];
		Nds[ty][tx] = Nd[(i * TILE_WIDTH + ty) * Width + col];
		__syncthreads();
		
		for (int j = 0; j < TILE_WIDTH; ++j)
			product += Mds[ty][j] * Nds[j][tx];
		__syncthreads();
	}
	Pd[row * Width + col] = product;
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
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);

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
