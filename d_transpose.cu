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
        //This is similar to above, but not exactly the same.
        //You'll need to use the blkDim and the tileSz parameters.
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
    //1) Use the blockIdx.y, blockDim.y and threadIdx.y
    //   to calculate the row of the input matrix
    //2) Use the blockIdx.x, blockDim.x and threadIdx.x
    //   to calculate the col of the input matrix
    //3) Flatten the row and col to determine an index into the input array
    //4) Flatten the col and row to determine an index into the result array
    //5) Do the tranpose

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
    //1) Use the blockIdx.y, blockDim.y, threadIdx.y, and tileSz
    //   to calculate the smallest row of a tile of the input matrix.
    //2) Use the blockIdx.x, blockDim.x and threadIdx.x, and tileSz
    //   to calculate the smallest col of a tile of the input matrix.
    //3) Loop through the tile of the input matrix. 
    //4) In the loop, flatten the current row and current col to determine an index 
    //   into the input array.
    //5) In the loop, flatten the current col and current row to determine an index 
    //   into the result array.
    //6) In the loop, copy the input element into the result array.

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


/*  
    d_transposeOptTiledKernel
    This kernel performs a optimized tiled transpose of an input matrix 
    and stores the result in the d_result matrix.
    Each matrix is of size width by width and has been linearized.
    Each thread performs the transpose of 16 elements.  
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
    //1) Use the blockIdx.y, blockDim.y, threadIdx.y, and OPTTILESZ
    //   to calculate the smallest row of a tile of the input matrix.
    //2) Use the blockIdx.x, blockDim.x and threadIdx.x, and OPTTILESZ
    //   to calculate the smallest col of a tile of the input matrix.
    //3) Loop through the tile of the input matrix, copying elements 
    //   from the input array into the tile array. Instead of copying 
    //   16 float values, copy 8 doubles (two doubles per row). You can do 
    //   this trick using double pointers. 
    //4) Do the transpose in the tile array (six swaps).
    //5) Use a loop to copy the elements in the tile array into the result 
    //   array.  Again, you'll use double pointers so that you end up 
    //   copying 8 doubles instead of 16 floats.

    //Of course like the previous kernels, you'll need to flatten the row and
    //column values.

    //Be careful to not access outside of the bounds of the
    //input and result matrices.

}      

