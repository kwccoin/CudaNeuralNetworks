# Target rules
# make -f Makefile
# make -f Makefile run

# can amend debug flag -DDEBUG1 etc. 
# make
# %k to clear terminal 
# make debug

all: build

build: nn-2_cuda.out nn-2_cuda.debug

nn-2_cuda.out:nn-2.cu nn-2_cuda.cu utils-2_cuda.cu utils-2.c nn-2.h nn-2_cuda.h
	nvcc nn-2.cu -o nn-2_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

nn-2_cuda.debug:nn-2.cu nn-2_cuda.cu utils-2_cuda.cu utils-2.c nn-2.h nn-2_cuda.h
	nvcc nn-2.cu -o nn-2_cuda.debug -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand -DDEBUG2c 
	# -DDEBUG2c

run: build
	nvprof ./nn-2_cuda.out

r2: build
	nvprof ./nn-2_cuda.out 2


debug: build
	nvprof ./nn-2_cuda.debug

d2: build
	nvprof ./nn-2_cuda.debug 2

clean:
	rm -f nn-2_cuda.out 
	rm -f nn-2_cuda.debug

clobber: clean

touch:
	touch *.cu *.h *.c
