NVCC = /usr/local/cuda-11.1/bin/nvcc
CC = g++

#Optimization flags. Don't use this for debugging.
NVCCFLAGS = -c -m64 -O2 --compiler-options -Wall -Xptxas -O2,-v

#No optimizations. Debugging flags. Use this for debugging.
#NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = wrappers.o transpose.o h_transpose.o d_transpose.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

transpose: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o transpose

transpose.o: transpose.cu h_transpose.h d_transpose.h 

h_transpose.o: h_transpose.cu h_transpose.h 

d_transpose.o: d_transpose.cu d_transpose.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm transpose *.o
