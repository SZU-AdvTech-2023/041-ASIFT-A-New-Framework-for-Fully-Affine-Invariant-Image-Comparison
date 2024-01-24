## Usage

### Command Syntax
```shell
ASIFT_CUDA [detectOption] [image1] [image2] [resultPath]
```


### Options
- `-origin` : Use original ASIFT.
- `-openmp` : Use original ASIFT with OpenMP for parallel processing.
- `-cudasift` : Replace SIFT in original ASIFT with CUDA-SIFT.
- `-openmp_cudasift` : Use CUDA-SIFT with OpenMP. (Image may be split due to insufficient GPU memory)
- `-rtg` : Run image rotate, tilt, and Gaussian blur on GPU.
- `-rtg_cudasift` : Run image processing on GPU using CUDA-SIFT.
- `-openmp_rtg_cudasift` : Image processing on GPU, using CUDA-SIFT and OpenMP. (Image may be split due to insufficient GPU memory)
- `-npp_cudasift` : Image processing on GPU using Nvidia 2D Image and Signal Performance Primitives with CUDA-SIFT.

### Parameters
- `[detectOption]` : Detection option as per above.
- `[image1]` : Path to the first image.
- `[image2]` : Path to the second image.
- `[resultPath]` : Path for saving the output.

### Example
```shell
ASIFT_CUDA -openmp_cudasift ./image1.jpg ./image2.jpg ./results/
```