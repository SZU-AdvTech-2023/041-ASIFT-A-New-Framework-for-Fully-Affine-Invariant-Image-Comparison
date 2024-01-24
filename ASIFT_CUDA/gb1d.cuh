#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<assert.h>
#include<math.h>

void GaussianBlur1D_gpu(float* image, int width, int height, float sigma, int flag_dir);

__global__ void ConvVertical_gpu(float* image, int width, int height, float* kernel, int ksize,float* buffer_t);

__device__ void ConvBufferFast_gpu(float* buffer, float* kernel, int rsize, int ksize);
