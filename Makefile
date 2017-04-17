# Target rules
# make -f Makefile
# make -f Makefile run

all: build

build: nn_cuda nn-2_cuda

nn_cuda:nn.cu parallel.cu utils.c
	nvcc nn.cu -o nn_cuda -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

nn-2_cuda:nn.cu parallel.cu utils.c
	nvcc nn-2.cu -o nn-2_cuda -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand


run: build
	nvprof ./nn_cuda

run-2: build
	nvprof ./nn-2_cuda

clean:
	rm -f nn_cuda
	rm -f nn-2_cuda
	
debug:
	nvcc nn.cu -o nn_cuda -DDEBUG -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand
	nvcc nn-2.cu -o nn-2_cuda -DDEBUG -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand


clobber: clean

touch:
	touch *.cu
