// see http://stackoverflow.com/questions/14818084/what-is-the-proper-include-for-the-function-sleep-in-c

//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>

#define WARP_SIZE 16
//#define DEBUG false
//#define DEBUG true

// use this and then if there is -DDEBUG it would be set but if not then it is false!

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef DEBUGU
#define DEBUGU false
#endif


#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

// should be 2 as cuda from non_cuda one

/* ---------------- [[HELPER FUNCTIONS FOR GLOBAL MEMORY]] ---------------- */

float *_copyHostDevice(float *src, int src_size) {
    float *src_d;
    cudaMalloc((void**)&src_d, sizeof(float) * src_size);
    cudaMemcpy(src_d, src, sizeof(float) * src_size, cudaMemcpyHostToDevice);
    return src_d;
}

float *_copyDeviceHost(float *src, int src_size, float *dst=NULL) {
    float *target;
    if (dst == NULL) {
        target = (float*)malloc(sizeof(float) * src_size);
    } else {
        target = dst;
    }

    cudaMemcpy(target, src, sizeof(float) * src_size, cudaMemcpyDeviceToHost);
    return target;
}

/* ---------------- [[HELPER FUNCTIONS FOR TILING]] ---------------- */

typedef struct {
    int x;
    int y;
} GlobalDim;

__device__ GlobalDim getGlobalDim(dim3 blockDim, dim3 blockIdx, dim3 threadIdx) {
    GlobalDim gd;
    gd.x = blockDim.x * blockIdx.x + threadIdx.x;
    gd.y = blockDim.y * blockIdx.y + threadIdx.y;
    return gd;
}

dim3 getGridBasedOnBlockSize(int width, int height, int block_size) {
    int gridX = (int)ceil((float)width / block_size);
    int gridY = (int)ceil((float)height / block_size);
    return dim3(gridX, gridY);
}

/* ---------------- [[HELPER FUNCTIONS FOR DEBUGGING]] ---------------- */

void _sleep(int n) {

  //sleep:
  #ifdef _WIN32
  Sleep(n);
  #else
  usleep(n*1000);  /* sleep for 100 milliSeconds */
  #endif
  
    //#ifdef __APPLE__
        //sleep(n);
    //#else _WIN32
        //sleep(n * 1000);
    //#endif
}

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("u-1-86 %f ", m[i * width + j]);
        }
        printf("\n");
    }
}
