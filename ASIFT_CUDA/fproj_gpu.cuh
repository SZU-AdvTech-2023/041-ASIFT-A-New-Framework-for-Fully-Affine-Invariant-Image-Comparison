#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include <stdio.h>
#include <math.h>
//#include "splines.cpp"
using namespace std;

float* fproj_gpu(float* in, float* out, int nx, int ny, int* sx, int* sy, float bg, int o, float p, char* i, float X1, float Y1, float X2, float Y2, float X3, float Y3, float* x4, float* y4);
__device__ void keys_gpu(float* c, float t, float a);
__device__ void spline3_gpu(float* c, float t);
__device__ float ipow_gpu(float x, int n);
__device__ void splinen_gpu(float* c, float t, float* a, int n);
__device__ float v_gpu(float* in, int x, int y, float bg, int width, int height);
void finvspline_gpu(float* In, int order, float* Out, int width, int height, double* buffer);
void init_splinen_gpu(float* a, int n);
__global__ void invspline1D_gpu(double* c, int size, double* z, int npoles, int total);
double initcausal_gpu(double* c, int n, double z);
double initanticausal_gpu(double* c, int n, double z);

__global__ void fproj_MT(float* in, float* out, float* ref, int nx, int ny, int sx, int sy, int o, float p, char* i, float bg, float xx, float yy, float a, float b, float x12, float x13, float y12, float y13, float X1, float Y1, int total);
