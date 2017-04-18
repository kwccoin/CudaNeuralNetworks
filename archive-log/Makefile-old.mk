# Target rules
# make -f Makefile
# make -f Makefile run

# make
# can amend debug flag -DDEBUG1 etc. 

all: build

build: nn-1_cuda.out nn-1_cuda.debug

nn-1_cuda.out:nn.cu parallel.cu utils.c 
	nvcc nn.cu -o nn-1_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

nn-1_cuda.debug:nn.cu parallel.cu utils.c 
	nvcc nn.cu -o nn-1_cuda.debug -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand -DDEBUG

run: build
	nvprof ./nn-1_cuda.out

debug:
	nvprof ./nn-1_cuda.debug

clean:
	rm -f nn-1_cuda.out 
	rm -f nn-1_cuda.debug

clobber: clean

touch:
	touch *.cu *.h *.c
