//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include "cudautils.cuh"
#include "cudaSift.h"

#include <thrust/extrema.h>

#ifndef __CUDACC__
#define __CUDACC__
#include "device_functions.h"
#endif

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define NUM_SCALES      5

// Scale down thread block width
#define SCALEDOWN_W    64 // 60 

// Scale down thread block height
#define SCALEDOWN_H    16 // 8

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H       8

// Find point thread block width
#define MINMAX_W       30 //32 

// Find point thread block height
#define MINMAX_H        8 //16 

// Laplace thread block width
#define LAPLACE_W     128 // 56

// Laplace rows per thread
#define LAPLACE_H       4

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

#endif



__global__ void ScaleDownDenseShift_gpu(float* d_Result, float* d_Data, int width, int pitch, int height, int newpitch, float* d_ScaleDownKernel);
__global__ void ScaleDownDense_gpu(float* d_Result, float* d_Data, int width, int pitch, int height, int newpitch, float* d_ScaleDownKernel);
__global__ void ScaleDown_gpu(float* d_Result, float* d_Data, int width, int pitch, int height, int newpitch, float* d_ScaleDownKernel);
__global__ void ScaleUp_gpu(float* d_Result, float* d_Data, int width, int pitch, int height, int newpitch);
__global__ void ExtractSiftDescriptors_gpu(cudaTextureObject_t texObj, SiftPoint* d_sift, int fstPts, float subsampling);
__device__ float FastAtan2_gpu(float y, float x);
__global__ void ExtractSiftDescriptorsCONSTNew_gpu(cudaTextureObject_t texObj, SiftPoint* d_sift, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void ExtractSiftDescriptorsCONST_gpu(cudaTextureObject_t texObj, SiftPoint* d_sift, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void ExtractSiftDescriptorsOld_gpu(cudaTextureObject_t texObj, SiftPoint* d_sift, int fstPts, float subsampling);
__device__ void ExtractSiftDescriptor_gpu(cudaTextureObject_t texObj, SiftPoint* d_sift, float subsampling, int octave, int bx);
__global__ void RescalePositions_gpu(SiftPoint* d_sift, int numPts, float scale);
__global__ void ComputeOrientations_gpu(cudaTextureObject_t texObj, SiftPoint* d_Sift, int fstPts, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void ComputeOrientationsCONSTNew_gpu(float* image, int w, int p, int h, SiftPoint* d_Sift, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void ComputeOrientationsCONST_gpu(cudaTextureObject_t texObj, SiftPoint* d_Sift, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void OrientAndExtractCONST_gpu(cudaTextureObject_t texObj, SiftPoint* d_Sift, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void FindPointsMultiTest_gpu(float* d_Data0, SiftPoint* d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void FindPointsMultiNew_gpu(float* d_Data0, SiftPoint* d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void FindPointsMulti_gpu(float* d_Data0, SiftPoint* d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void FindPointsMultiOld_gpu(float* d_Data0, SiftPoint* d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
__global__ void LaplaceMultiTex_gpu(cudaTextureObject_t texObj, float* d_Result, int width, int pitch, int height, int octave, float* d_LaplaceKernel);
__global__ void LaplaceMultiMem_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, int octave, float* d_LaplaceKernel);
__global__ void LaplaceMultiMemWide_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, int octave, float* d_LaplaceKernel);
__global__ void LaplaceMultiMemTest_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, int octave, float* d_LaplaceKernel);
__global__ void LaplaceMultiMemOld_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, int octave, float* d_LaplaceKernel);
__global__ void LowPass_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, float* d_LowPassKernel);
__global__ void LowPassBlockOld_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, float* d_LowPassKernel);
__global__ void LowPassBlock_gpu(float* d_Image, float* d_Result, int width, int pitch, int height, float* d_LowPassKernel);