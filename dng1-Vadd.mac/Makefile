# ubuntu same Makefile as mac
# Target rules
# make -f Makefile
# make -f Makefile run

# can amend debug flag -DDEBUG1 etc. 
# make
# %k to clear terminal 
# make debug

# sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer

#sudo xcode-select --switch /Library/Developer/CommandLineTools  

#pkgutil --pkg-info=com.apple.pkg.CLTools_Executables

#clang --version

#Need re-boot … for 8.2 at least but not working trying 7.3 ... 



all: build

build: dn-Vadd.out dn-Vadd_cuda.out

dn-Vadd.out:dn-Vadd.c
	nvcc dn-Vadd.c -o dn-Vadd.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

dn-Vadd_cuda.out:dn-Vadd.cu
	nvcc dn-Vadd.cu -o dn-Vadd_cuda.out -DGPU -I/usr/local/cuda/include/ -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand

run: build
	nvprof ./dn-Vadd_cuda.out 512

clean:
	rm -f *.out 
	rm -f *.debug

clobber: clean

touch:
	touch *.cu *.h *.c
