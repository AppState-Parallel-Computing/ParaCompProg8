NVCC = /usr/bin/nvcc
CC = g++

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX
NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

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
