# Target rules
# make -f Makefile
# make -f Makefile run

all: build

build: nn_cuda

nn_cuda:nn.cu parallel.cu utils.c
	nvcc nn.cu -o nn_cuda -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

run: build
	nvprof ./nn_cuda

clean:
	rm -f nn_cuda

clobber: clean
