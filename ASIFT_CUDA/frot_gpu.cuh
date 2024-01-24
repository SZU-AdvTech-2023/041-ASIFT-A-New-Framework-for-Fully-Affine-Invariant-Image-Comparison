#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "library.h"
#include <vector>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
using namespace std;

float* frot_gpu(float* in, float* out, int nx, int ny, int* nx_out, int* ny_out, float a, float b, char* k_flag);
__host__ __device__ void bound_gpu(int x, int y, float ca, float sa, int* xmin, int* xmax, int* ymin, int* ymax);
__global__ void frot_MT(float* in, float* out, int sx, int sy, int xmin, int ymin, float ca, float sa, float xtrans, float ytrans, int nx, int ny, float b, int total);
