# Target rules
# make -f Makefile
# make -f Makefile run

# make
# can amend debug flag -DDEBUG1 etc. 

all: build

build: nn-2_cuda.out nn-2_cuda.debug

nn-2_cuda.out:nn-2.cu nn-2_cuda.cu utils-2.c autopsy-2_cuda.c nn-2.h nn-2_cuda.h
	nvcc nn-2.cu -o nn-2_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

nn-2_cuda.debug:nn-2.cu nn-2_cuda.cu utils-2.c autopsy-2_cuda.c nn-2.h nn-2_cuda.h
	nvcc nn-2.cu -o nn-2_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand -DDEBUG

run: build
	nvprof ./nn-2_cuda.out

debug:
	nvprof ./nn-2_cuda.debug

clean:
	rm -f nn-2_cuda.out 
	rm -f nn-2_cuda.debug

clobber: clean

touch:
	touch *.cu *.h *.c
