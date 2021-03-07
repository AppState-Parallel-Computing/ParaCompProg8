#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include "h_transpose.h"
#include "d_transpose.h"
#include "wrappers.h"

//If you set this to 1, the program will print the input and output arrays.
//You probably don't want to do this with a big matrix.
#define DEBUG 0

//default values for parameters
#define WHICH_DEFAULT NAIVE
#define BLKDIM_DEFAULT 8
#define MATDIM_DEFAULT 11
#define TILESZ_DEFAULT 1
#define BLKDIMARR 3
#define TILESZARR 5
#define MATDIMARR 11

//legal dimensions for the block, tile, and matrix
static int blkDims[BLKDIMARR] = {8, 16, 32};
static int tileSzs[TILESZARR] = {1, 2, 4, 8, 16};
static int matDims[MATDIMARR] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

//prototypes for functions in this file
static void initMatrix(float * array, int length);
static void parseArgs(int argc, char * argv[], int * matrixDim, int * blkDim, 
               int * tileSz, int * which, bool * doTime);
static int check(const char * label, int * values, int len, int what);
static void compare(float * result1, float * result2, int width);
static void printUsage();
static void printHeaders(int matrixDim, int blkDim, int tileSz, int which);

//helper function for debugging
static void printMatrix(const char * header, float * matrix, int width);

/*
   driver for the transpose program.  
*/
int main(int argc, char * argv[])
{
    //parameters for GPU version
    int matrixDim, blkDim, tileSz, which;
    bool doTime;

    //parse the command line arguments
    parseArgs(argc, argv, &matrixDim, &blkDim, &tileSz, &which, &doTime);

    //dynamically allocate the matrices
    float * h_input = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float * h_result = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float * d_result = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);

    //CPU time, GPU time, speedup
    float h_time, d_time, speedup;

    //initialize matrices
    initMatrix(h_input, matrixDim * matrixDim);
    memset(h_result, 0, matrixDim * matrixDim);
    memset(d_result, 0, matrixDim * matrixDim);

    //print headers
    printHeaders(matrixDim, blkDim, tileSz, which);
   
    //perform the transpose of the matrix on the CPU
    h_time = h_transpose(h_input, h_result, matrixDim);

    if (DEBUG)
    {
        printMatrix("Input", h_input, matrixDim);
        printMatrix("CPU Result", h_result, matrixDim);
    }

    //perform the transpose of the matrix on the GPU
    d_time = d_transpose(d_result, h_input, matrixDim, blkDim, tileSz, which);
    if (DEBUG)
        printMatrix("GPU Result", d_result, matrixDim);

    //compare GPU and CPU results 
    compare(h_result, d_result, matrixDim);
    printf("GPU result is correct.\n");

    //run the GPU version multiple times to get somewhat accurate timing
    if (doTime == true)
    {
        //Because the GPU time varies greatly, we will run the GPU code
        //multiple times and compute an average.
        //In addition, ignore the first couple of times since it takes
        //time for the GPU to "warm-up."
        printf("Timing the kernel. This may take a bit.\n");
        d_time = d_transpose(d_result, h_input, matrixDim, blkDim, tileSz, which);
        d_time = 0;
        int i;
        for (i = 0; i < 5; i++)
            d_time += d_transpose(d_result, h_input, matrixDim, blkDim, tileSz, which);
        d_time = d_time/5.0;

        //Output the times and the speedup
        printf("\nTiming\n");
        printf("------\n");
        printf("CPU: \t\t\t%f msec\n", h_time);
        printf("GPU: \t\t\t%f msec\n", d_time);
        speedup = h_time/d_time;
        printf("Speedup: \t\t%f\n", speedup);
    }

    //free dynamically allocated data
    free(h_result);
    free(d_result);
    free(h_input);
}    

/*
 * printHeaders
 * Output information about the tranpose code that will be executed.
 * matrixDim - size of matrix to be transposed (matrixDim by matrixDim)
 * blkDim - size of the thread blocks executed on GPU (blkDim by blkDim)
 * tileSz - size of the tile of elements handled by one thread on 
 *          the GPU (tileSz by tileSz)
 * which - which GPU kernel is executed (NAIVE, TILED, OPTTILED)
*/
void printHeaders(int matrixDim, int blkDim, int tileSz, int which)
{

    printf("Transpose of matrix of size %d by %d.\n", matrixDim, matrixDim);
    if (which == NAIVE)
    {
        printf("Using naive kernel with ");
        printf("%d by %d block.\n", blkDim, blkDim);
    } else if (which == TILED)
    {
        printf("Using tiled kernel with ");
        printf("%d by %d block and ", blkDim, blkDim);
        printf("%d by %d tile.\n", tileSz, tileSz);
    } else if (which == OPTTILED)
    {
        printf("Using optimized tiled kernel with ");
        printf("%d by %d block and ", blkDim, blkDim);
        printf("%d by %d tile.\n", tileSz, tileSz);
    }
}

/* 
    parseArgs
    This function parses the command line arguments to get
    the dimension of the matrices, the size of the thread blocks,
    the size of the tile of elements handled by one thread, the GPU
    kernel to be executed, and whether timing results should be
    produced. If the command line argument is invalid, it prints usage 
    information and exits.
    Inputs:
    argc - count of the number of command line arguments
    argv - array of command line arguments
    matrixDimP - pointer to an int to be set to the matrix dimensions
    blkDimP - pointer to an int to be set to the block dimensions
    tileSzP - pointer to an int to be set to the size of the tile
              of elements to be handled by one thread
    whichP - which kernel to execute
    doTimeP - pointer to a bool that is set to true or false if timing
              is to be performed
*/
void parseArgs(int argc, char * argv[], int * matrixDimP,
               int * blkDimP, int * tileSzP, int * whichP, bool * doTimeP)
{
    int i;
    //set the parameters to their defaults
    int dimExp = MATDIM_DEFAULT;
    int blkDim = BLKDIM_DEFAULT;
    int tileSz = TILESZ_DEFAULT;
    int which = WHICH_DEFAULT;
    bool doTime = false;

    //loop through the command line arguments
    for (i = 1; i < argc; i++)
    {
       if (i < argc - 1 && strcmp(argv[i], "-n") == 0)
       {
          dimExp = atoi(argv[i+1]);
          i++;   //skip over the argument after the -n
       }
       else if (i < argc - 1 && strcmp(argv[i], "-b") == 0)
       {
          blkDim = atoi(argv[i+1]);
          i++;   //skip over the argument after the -b
       }
       else if (i < argc - 1 && strcmp(argv[i], "-t") == 0)
       {
          tileSz = atoi(argv[i+1]);
          i++;   //skip over the argument after the -t
       }
       else if (strcmp(argv[i], "-naive") == 0)
          which = NAIVE;
       else if (strcmp(argv[i], "-tiled") == 0)
          which = TILED;
       else if (strcmp(argv[i], "-opttiled") == 0)
          which = OPTTILED;
       else if (strcmp(argv[i], "-time") == 0)
          doTime = true;
       else
          printUsage();
    }

    //check if the provided parameters are correct
    if (!check("block dimensions", blkDims, BLKDIMARR, blkDim)) printUsage();
    if (!check("matrix dimensions", matDims, MATDIMARR, dimExp)) printUsage();
    if (!check("tile dimensions", tileSzs, TILESZARR, tileSz)) printUsage();
    if (which == OPTTILED) tileSz = 4;  //ignore user input
    if (which == NAIVE) tileSz = 1;     //ignore user input

    (*blkDimP) = blkDim;
    (*tileSzP) = tileSz;
    (*matrixDimP) = 1 << dimExp;
    (*whichP) = which;
    (*doTimeP) = doTime;
}

/*
 * check
 * Takes as input an array of size len containing ints and
 * the int value what.  If what is in the array, the function returns 1.
 * Otherwise, the function prints an error message and returns 0.
 * This function is used to check command line arguments. A valid
 * input can be found in the values array. 
 * Inputs
 * label - printed with the error message
 * values - array of legal values
 * len - length of array 
 * what - value to search for in values array
 * Returns
 * 1 - legal what value
 * 0 - illegal what value
*/
int check(const char * label, int * values, int len, int what)
{
   int i;
   for (i = 0; i < len; i++)
      if (values[i] == what) return 1;
   printf("Invalid %s - valid values are: ", label);
   for (i = 0; i < len; i++) printf("%d ", values[i]);
   printf("\n");
   return 0;
}

/*
    printUsage
    prints usage information and exits
*/
void printUsage()
{
    printf("\nThis program performs the transpose of an array\n");
    printf("of size n by n using the CPU and one of three CUDA routines\n");
    printf("usage: transpose [-n <n> | -b <b> | -t <t> | -naive | -tiled | -opttiled | -time] \n");
    printf("       2**<n> is the height and width of the matrix\n");
    printf("              default <n> is %d\n", MATDIM_DEFAULT);
    printf("       <b> by <b> is the size of each block of threads\n");
    printf("              default <b> is %d\n", BLKDIM_DEFAULT);
    printf("       <t> by <t> is the size of the tile of elements handled by one thread\n");
    printf("              default <t> is %d\n", TILESZ_DEFAULT);
    printf("       -naive use the naive CUDA version\n");
    printf("       -tiled use the tiled CUDA version\n");
    printf("       -opttiled use the optimized tiled CUDA version\n");
    printf("       -time implementation and output speedup\n");
    exit(EXIT_FAILURE);
}

/* 
    initMatrix
    Initializes an array of floats of size
    length to random values between 0 and 100.
    Inputs:
    array - pointer to the array to initialize
    length - length of array
*/
void initMatrix(float * array, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        int randVal = rand();
        float frac = (randVal % 10)/10.0;
        randVal = randVal % 100;
        array[i] = (float) randVal + frac;
    }
}

/*
    compare
    Compares the values in two matrices and outputs an
    error message and exits if the values do not match.
    Inputs
    result1, result2 - float matrices
    n - dimension of each matrix is n by n
    label - string to use in the output message if an error occurs
*/
void compare(float * result1, float * result2, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        { 
            float diff = abs(result1[i * n + j] - result2[i * n + j]);
            if (diff > 0) // 
            {
                printf("GPU transpose does not match CPU results.\n");
                printf("cpu result[%d, %d]: %f, gpu: result[%d, %d]: %f\n", 
                   i, j, result1[i * n + j], i, j, result2[i * n + j]);
                exit(EXIT_FAILURE);
            }
        }
    }
}

//helper function for debugging
/*
 * printMatrix
 * Outputs the header and a matrix. Matrix is of size width by width
*/
void printMatrix(const char * header, float * matrix, int width)
{
    int i, j;
    printf("\n%s:\n", header);
    for (i = 0; i < width; i++)
    {
        for (j = 0; j < width; j++)
        {
            printf("%4.1f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

