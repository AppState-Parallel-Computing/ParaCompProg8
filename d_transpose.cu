#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "helpers.h"
#include "d_transpose.h"

//tile size for optimized kernel must be 4
#define OPTTILESZ 4

//prototypes for kernels in this file
__global__ 
void d_transposeNaiveKernel(float * d_result, float * d_input, int width);

__global__ 
void d_transposeTiledKernel(float * d_result, float * d_input, int width, int tileSz);

__global__ 
void d_transposeOptTiledKernel(float * d_result, float * d_input, int width);

/*  d_transpose
    This function prepares and invokes a kernel to perform
    a matrix transpose on the GPU. The matrices have been 
    linearized so each array is 1D and contains width * width elements.
    Inputs:
    result - points to a matrix to hold the transposed result
    input - points to the input matrix 
    width - width and height of the input and result matrices
    blkDim - dimensions of each block of threads to be launched
    tileSz - dimension of the size of a tile of elements to be handled
             by one thread (1 for NAIVE version, 4 for OPTTILED,
             TILED version can be 1, 2, 4, 8, or 16)
    which - indicates which kernel to use (NAIVE, TILED, OPTTILED)
*/
float d_transpose(float * result, float * input, int width, int blkDim, 
                  int tileSz, int which)
{
    float * d_result, * d_input;  //pointers to matrices for GPU
   
    //CUERR is a macro in helpers.h that checks for a Cuda error 
    //Begin the timing (macro in helpers.h) 
    TIMERSTART(gpuTime)
    //Allocate space in GPU memory for input matrix
    cudaMalloc((void **)&d_input, sizeof(float) * width * width);            CUERR
    //Copy input from CPU memory to GPU memory
    cudaMemcpy(d_input, input, sizeof(float) * width * width, H2D);          CUERR
    //Allocate space in GPU memory for result matrix
    cudaMalloc((void **)&d_result, sizeof(float) * width * width);           CUERR

    //Launch the appropriate kernel
    if (which == NAIVE)
    {
        //Define the block and the grid and launch the naive kernel
        int grdDim = SDIV(width, blkDim); 
        dim3 block(blkDim, blkDim, 1);
        dim3 grid(grdDim, grdDim, 1);
        d_transposeNaiveKernel<<<grid, block>>>(d_result, d_input, width);   CUERR
    } else if (which == TILED)
    {
        //TO DO
        //Define the block and the grid and launch the tiled kernel
        //Be careful to not define a grid that is too big
    } else if (which == OPTTILED)
    {
        //TO DO
        //Define the block and the grid and launch the optimized tiled kernel
        //Be careful to not define a grid that is too big
    }
    
    //wait for threads to finish
    cudaDeviceSynchronize();                                                  CUERR
    //copy result from GPU memory to CPU memory
    cudaMemcpy(result, d_result, sizeof(float) * width * width, D2H);         CUERR

    //free dynamically  allocated memory
    cudaFree(d_result);                                                       CUERR
    cudaFree(d_input);                                                        CUERR

    //stop the timer
    TIMERSTOP(gpuTime)
    return TIMEELAPSED(gpuTime);
}

/*  
    d_transposeNaiveKernel
    This kernel performs a naive transpose of an input matrix 
    and stores the result in the d_result matrix.
    Each matrix is of size width by width and has been linearized.
    Each thread performs the transpose of element.  
    Inputs:
    d_result - pointer to the array in which the result is stored
    d_input - pointer to the array containing the input
    width - width and height of the matrices
*/
__global__ 
void d_transposeNaiveKernel(float * d_result, float * d_input, int width) 
{
    //TO DO
    //You need to use the block and thread identifiers, and the
    //block dimensions to calculate the row and the col of the
    //input element to transpose. Then, you'll take the row and
    //col and flatten those values to index into the d_input and
    //d_result arrays.
    //Be careful to not access outside of the dimensions of the arrays.
}      

/*  
    d_transposeTiledKernel
    This kernel performs a tiled transpose of an input matrix 
    and stores the result in the d_result matrix.
    Each matrix is of size width by width and has been linearized.
    Each thread performs the transpose of tile by tile elements.  
    Inputs:
    d_result - pointer to the array in which the result is stored
    d_input - pointer to the array containing the input
    width - width and height of the matrices
*/
__global__ 
void d_transposeTiledKernel(float * d_result, float * d_input,
                            int width, int tileSz) 
{

    //TO DO
    //You'll need to use the block and thread identifiers, the block
    //dimensions, and the tile size to calculate the row and the
    //column of the first element in the tile to transpose.
    //Be careful to not access outside of the bounds of the
    //input and result matrices.
}      

/*
 * swap
 * Swap the contents of two floats in the device memory.
 * Inputs
 * fval1 - pointer to one of the floats
 * fval2 - pointer to the other float
 * Result
 * (*fval1) and (*fval2) values are swapped
*/
__device__
void swap(float * fval1, float * fval2)
{
   float tmp;
   tmp = (*fval1);
   (*fval1) = (*fval2);
   (*fval2) = tmp;
}

#define PRINT 0
//#define PRINT (threadIdx.x == 1 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0)
//#define PRINT (row == 0 && col == 4)

/*  
    d_transposeOptTiledKernel
    This kernel performs a optimized tiled transpose of an input matrix 
    and stores the result in the d_result matrix.
    Each matrix is of size width by width and has been linearized.
    Each thread performs the transpose of 4  by 4 elements.  
    Inputs:
    d_result - pointer to the array in which the result is stored
    d_input - pointer to the array containing the input
    width - width and height of the matrices
*/
__global__ 
void d_transposeOptTiledKernel(float * d_result, float * d_input, int width)
{
    float tile[OPTTILESZ][OPTTILESZ];

    //TO DO
    //You'll need to use the block and thread identifiers, the block
    //dimensions, and the tile size (4) to calculate the row and the
    //column of the first element in the tile to transpose.
    //Be careful to not access outside of the bounds of the
    //input and result matrices.

    //Copy the appropriate elements into the tile array.
    //Then perform the transpose in the tile array (six swaps).
    //Finally, copy the transposed elements to the results
}      

