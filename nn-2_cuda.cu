#include <math.h>

// http://stackoverflow.com/questions/17076956/nvcc-linking-error-in-c-and-cuda-c-code
#include "nn-2_cuda.h"
#include "nn-2.h"

#include "utils-2.c"

#include "utils-2_cuda.cu"



/* ---------------- [[CUDA KERNELS]] ---------------- */

__global__ void updateWeights_CUDA(float *weights, float *changes, float *delta_outputs, float *inputs, int n_inputs, int n_outputs) {
    int width = n_outputs;
    int height = n_inputs;
    GlobalDim gd = getGlobalDim_CUDA(blockDim, blockIdx, threadIdx);

    if ((gd.x < width) && (gd.y < height)) {
        int idx = width * gd.y + gd.x;
        float change = delta_outputs[gd.x] * inputs[gd.y];
        
        weights[idx] += 0.5 * change + 0.5 * changes[idx]; // 0.5 is magic no. use two 1.0 instead
        changes[idx] = change;
    }

}

__global__ void mapStep_CUDA(float *inputs, float *matrix, float *buffer, int width, int height) {
    GlobalDim gd = getGlobalDim_CUDA(blockDim, blockIdx, threadIdx);

    if ((gd.x < width) && (gd.y < height)) {
        int idx = width * gd.y + gd.x;
        buffer[idx] = inputs[gd.y] * matrix[idx];
    }
}

__global__ void reduceStep_CUDA(float *input, float *output, int width, int height) {

    __shared__ float sharedMemory[WARP_SIZE * WARP_SIZE];

    // STEP 1: exclude all threads that do not depend from problem
    GlobalDim gd = getGlobalDim_CUDA(blockDim, blockIdx, threadIdx);


    if ((gd.x < width) && (gd.y < height)) {

        // STEP 2: Move to shared memory
        int gridId = gd.y * width + gd.x;
        int blockId = threadIdx.y * blockDim.x + threadIdx.x;
        sharedMemory[blockId] = input[gridId];
        __syncthreads();

        int n = (int)ceil((float)blockDim.y/2);
        while(n >= 1) {
            if (threadIdx.y < n) {

                if ((gd.y + n) < height) {
                    int firstIndex = blockId;
                    int secondIndex = blockDim.x * (threadIdx.y + n) + threadIdx.x;
                    sharedMemory[firstIndex] += sharedMemory[secondIndex];
                }
            }
            __syncthreads();
            if (n == 1) {
                break;
            } else {
                n = (int)ceil((float)n/2);
            }
        }
        __syncthreads();

        // STEP 3: Write back results
        if (threadIdx.y == 1) {
            output[blockIdx.y * width + gd.x] = sharedMemory[threadIdx.x];
        }
    }
}

/* ---------------- [[LAUNCH FUNCTIONS]] ---------------- */

void setWeightsForLayers_CUDA(float *weights, float *changes, float *delta_outputs, float *inputs, int n_inputs, int n_outputs) {

    // Copy to device memory
    int grid_size = n_inputs * n_outputs;
    float *weights_d = _copyHostDevice_CUDA(weights, grid_size);
    float *changes_d = _copyHostDevice_CUDA(changes, grid_size);
    float *delta_outputs_d = _copyHostDevice_CUDA(delta_outputs, n_outputs);
    float *inputs_d = _copyHostDevice_CUDA(inputs, n_inputs);

    // Define block structure
    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid = getGridBasedOnBlockSize_CUDA(n_outputs, n_inputs, WARP_SIZE);

    // RUN RUN RUN!
    updateWeights_CUDA<<<grid, block>>>(weights_d, changes_d, delta_outputs_d, inputs_d, n_inputs, n_outputs);

    // Copy back weights and momenutm
    weights = _copyDeviceHost_CUDA(weights_d, grid_size, weights);
    changes = _copyDeviceHost_CUDA(changes_d, grid_size, changes);
}

// at least consistent with Cuda ending

void update_layer_CUDA(float *src_layer, float *dst_layer, int src_n, int dst_n, float *weights) {
    dim3 block(WARP_SIZE, WARP_SIZE);

    float *src_layer_d, *weights_d, *buffer_d;
    int total = src_n * dst_n;
 
    // Allocate input in global memory
    src_layer_d = _copyHostDevice_CUDA(src_layer, src_n);
    weights_d = _copyHostDevice_CUDA(weights, total);
    cudaMalloc((void**)&buffer_d, sizeof(float) * total);
 
    // Create block dimensions and run parallel update layer
    int gridX = (int)ceil((float)dst_n/WARP_SIZE);
    int gridY = (int)ceil((float)src_n/WARP_SIZE);
    dim3 grid(gridX, gridY);

    // RUN RUN RUN!
    if (DEBUGP) {
        printf("\n par-1-123 ***** Updating layer *****\n");

        printf("\n par-2-125 From drawMatrix(src_layer, src_n, 1\n");
        drawMatrix(src_layer, src_n, 1);

        printf("\nT par-3-128 o drawMatrix(weights, dst_n, src_n)\n");
        drawMatrix(weights, dst_n, src_n);
    }
    mapStep_CUDA<<<grid, block>>>(src_layer_d, weights_d, buffer_d, dst_n, src_n);

    // Set the current target to the input
    float *currentTarget = buffer_d;
    int currentHeight = src_n;

    while (currentHeight > 1) {

        // Calculate grid size
        int gridX = (int)ceil((float)dst_n/WARP_SIZE);
        int gridY = (int)ceil((float)currentHeight/WARP_SIZE);
        dim3 grid(gridX, gridY);

        // Allocate new buffer
        float *buffer_d;
        cudaMalloc((void**)&buffer_d, sizeof(float) * (dst_n * gridY));
 
        // RUN RUN RUN!
        reduceStep_CUDA<<<grid, block>>>(currentTarget, buffer_d, dst_n, currentHeight);

        // Free old memory and keep track of the new one
        cudaFree(currentTarget);
        currentHeight = grid.y;
        currentTarget = buffer_d;
    }

    dst_layer =_copyDeviceHost_CUDA(currentTarget, dst_n, dst_layer);
    for (int i=0; i < dst_n; i++) {
        dst_layer[i] = sigmoid(dst_layer[i]); // tanh(dst_layer[i]);  // just apply tanh???
    }

    if (DEBUGP) {
        printf("\n par-4-163 Result is drawMatrix(dst_layer, dst_n, 1) \n");
        drawMatrix(dst_layer, dst_n, 1);
        printf("\n par-5-165 ***** ENDED UPDATING LAYER *****\n");
        _sleep(1);
    }
}

