#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

/* must use .cu otherwise .c and .cpp will send to host compiler and global would have issues */
/* under w10 */

__global__ void VectorAdd(int *a, int *b, int *c, int n) {
	int i = threadIdx.x;
	// no loop for (i = 0; i < n; ++i)
	if (i < n)
		c[i] = a[i] + b[i];
}

int main(int argc, char *argv[])
{

	int noOfRun;
	if (argc > 1)
	{
		noOfRun = atoi(argv[1]);
		printf("\nargv[1] in intger=%d\n\n", noOfRun);
	}

	// use SIZE here instead of noofRun

	int *a, *b, *c;

	a = (int *)malloc(SIZE * sizeof(int));
	b = (int *)malloc(SIZE * sizeof(int));
	c = (int *)malloc(SIZE * sizeof(int));

	int *d_a, *d_b, *d_c;

	cudaMalloc(&d_a, SIZE * sizeof(int));
	cudaMalloc(&d_b, SIZE * sizeof(int));
	cudaMalloc(&d_c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i + 1;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	VectorAdd << <1, SIZE >> >(d_a, d_b, d_c, SIZE);

	cudaMemcpy(a, d_a, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; ++i)
		printf("host = %d: a[%d] + b[%d] = %d + %d = c[%d] = %d\n", i, i, i, a[i], b[i], i, c[i]);

	/* you cannot directly address the gpu memory !!!
	for (int i = 0; i < 10; ++i)
	printf("device = %d: d_a[%d] + d_b[%d] = %d + %d = d_c[%d] = %d\n", i, i, i, d_a[i], d_b[i], i, d_c[i]); */

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// cudaProfilerStop(); and _syncthreads(); and device level close ????

	return 0;
}

/*

#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

void VectorAdd(int *a, int *b, int *c, int n) {
int i;
for (i = 0; i < n; ++i)
c[i] = a[i] + b[i];
}

int main(int argc, char *argv[])
{

int noOfRun;
if (argc > 1)
{
noOfRun = atoi(argv[1]);
printf("\nargv[1] in intger=%d\n\n", noOfRun);
}

// use SIZE here instead of noofRun

int *a, *b, *c;

a = (int *)malloc(SIZE * sizeof(int));
b = (int *)malloc(SIZE * sizeof(int));
c = (int *)malloc(SIZE * sizeof(int));

for (int i = 0; i < SIZE; ++i)
{
a[i] = i;
b[i] = i + 1;
c[i] = 0;
}

VectorAdd(a, b, c, SIZE);

for (int i = 0; i < 10; ++i)
printf("%d: a[%d] + b[%d] = %d + %d = c[%d] = %d\n", i, i, i, a[i], b[i], i, c[i]);

free(a);
free(b);
free(c);

return 0;
}


*/

/*

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}

int main()
{
const int arraySize = 5;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
int c[arraySize] = { 0 };

// Add vectors in parallel.
cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "addWithCuda failed!");
return 1;
}

printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
c[0], c[1], c[2], c[3], c[4]);

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceReset failed!");
return 1;
}

return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
int *dev_a = 0;
int *dev_b = 0;
int *dev_c = 0;
cudaError_t cudaStatus;

// Choose which GPU to run on, change this on a multi-GPU system.
cudaStatus = cudaSetDevice(0);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
goto Error;
}

// Allocate GPU buffers for three vectors (two input, one output)    .
cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

// Copy input vectors from host memory to GPU buffers.
cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

// Launch a kernel on the GPU with one thread for each element.
addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
goto Error;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
goto Error;
}

// Copy output vector from GPU buffer to host memory.
cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

Error:
cudaFree(dev_c);
cudaFree(dev_a);
cudaFree(dev_b);

return cudaStatus;
}
*/
