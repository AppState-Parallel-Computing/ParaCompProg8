#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "helpers.h"
#include "h_transpose.h"

//prototype for function local to this file
void transposeOnCPU(float * h_input, float * h_result, int width);

/*  h_transpose
    This function uses the CPU to perform a transpose of the h_input array 
    and stores the result in the h_result array. It returns the amount of time 
    it takes to perform the transpose.
    Inputs:
    h_input - points to the input matrix for the transpose
    h_result - points to the matrix to hold the result
    width - x and y dimension of the matrices (width by width)

    returns the amount of time it takes to perform the matrix transpose
*/
float h_transpose(float * h_input, float * h_result, int width) 
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //CUERR is a macro defined in helpers.h that checks for a CUDA error
    //Use CUDA functions to do the timing 
    //Create event objects
    cudaEventCreate(&start_cpu);                       CUERR
    cudaEventCreate(&stop_cpu);                        CUERR
    //record the starting time
    cudaEventRecord(start_cpu);                        CUERR
    
    //call function that does the actual work
    transposeOnCPU(h_input, h_result, width);
   
    //record the ending time and wait for event to complete
    cudaEventRecord(stop_cpu);                              CUERR
    cudaEventSynchronize(stop_cpu);                         CUERR

    //calculate the elapsed time between the two events 
    cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu); CUERR 
    return cpuMsecTime;
}

/*  transposeOnCPU
    This function performs the matrix transpose on the CPU.  
    Inputs:
    h_input - points to the matrix to transpose
    h_result - points to the matrix to hold the result
    width - both the x and y dimension of the matrices (width by width)

    modifies the h_result matrix
*/
void transposeOnCPU(float * h_input, float * h_result, int width)
{
    int i, j; 
    for (i = 0; i < width; i++)
    {
        for (j = 0; j < width; j++)
        {
            h_result[j * width + i] = h_input[i * width + j];
        }
    }
}
