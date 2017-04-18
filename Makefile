# Target rules
# make -f Makefile
# make -f Makefile run

all: build

build: nn-2_cuda

nn-2_cuda:nn-2.cu nn-2_cuda.cu utils-2.c autopsy-2_cuda.c nn-2.h nn-2_cuda.h
	nvcc nn-2.cu -o nn-2_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand


run: build
	nvprof ./nn-2_cuda.out

clean:
	rm -f nn-2_cuda*
	
debug:

	nvcc nn-2.cu -o nn-2_cuda -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand -DDEBUG1


clobber: clean

touch:
	touch *.cu *.h *.c
