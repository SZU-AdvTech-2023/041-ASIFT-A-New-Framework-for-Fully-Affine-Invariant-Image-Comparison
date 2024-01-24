#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.cuh"
#include "cudaImage.h"
#include "cudaSift.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  


int ExtractSiftLoop(SiftData& siftData, CudaImage& img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float* memoryTmp, float* memorySub, int d_MaxNumPoints, float* d_LaplaceKernel, unsigned int* d_PointCounter);
void ExtractSiftOctave(SiftData& siftData, CudaImage& img, int octave, float thresh, float lowestScale, float subsampling, float* memoryTmp, int d_MaxNumPoints, float* d_LaplaceKernel, unsigned int* d_PointCounter);
double ScaleDown(CudaImage& res, CudaImage& src, float variance);
double ScaleUp(CudaImage& res, CudaImage& src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage& src, SiftData& siftData, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData& siftData, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData& siftData, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);
double RescalePositions(SiftData& siftData, float scale);
double LowPass(CudaImage& res, CudaImage& src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float* kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage& baseImage, CudaImage* results, int octave, float* d_LaplaceKernel);
double FindPointsMulti(CudaImage* sources, SiftData& siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, int d_MaxNumPoints, unsigned int* d_PointCounter);

#endif
