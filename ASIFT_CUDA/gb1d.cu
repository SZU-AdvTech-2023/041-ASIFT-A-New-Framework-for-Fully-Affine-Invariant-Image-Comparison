#include "gb1d.cuh"

const float GaussTruncate1_gpu = 4.0;

void GaussianBlur1D_gpu(float* image, int width, int height, float sigma, int flag_dir)
{
	clock_t start=clock(), end;

	float x, kernel[100], sum = 0.0;
	int ksize, i;

	/* The Gaussian kernel is truncated at GaussTruncate sigmas from
	center.  The kernel size should be odd.
	*/
	ksize = (int)(2.0 * GaussTruncate1_gpu * sigma + 1.0);
	ksize = max(3, ksize);    /* Kernel must be at least 3. */
	if (ksize % 2 == 0)       /* Make kernel size odd. */
		ksize++;
	assert(ksize < 100);

	/* Fill in kernel values. */
	for (i = 0; i <= ksize; i++) {
		x = float(i - ksize / 2);
		kernel[i] = exp(-x * x / (2.f * sigma * sigma));
		sum += kernel[i];
	}
	/* Normalize kernel values to sum to 1.0. */
	for (i = 0; i < ksize; i++)
		kernel[i] /= sum;

	float* kernel_gpu;
	cudaMalloc((float**)&kernel_gpu, 100 * sizeof(float));
	cudaMemcpy(kernel_gpu, kernel, 100 * sizeof(float), cudaMemcpyHostToDevice);

	if (flag_dir == 0)
	{
		//ConvHorizontal(image, width, height, kernel, ksize);
		return;
	}
	else
	{
		dim3 grid(1, 1), block(32, 32);
		if (width > 1024)
		{
			int gsize = sqrt(width) / 32 + 1;
			grid = dim3(gsize, gsize);
		}
		float* buffer_t;
		cudaMalloc((float**)&buffer_t, width * 10000 * sizeof(float));
		ConvVertical_gpu << <grid, block >> > (image, width, height, kernel_gpu, ksize,buffer_t);
		cudaError_t err = cudaGetLastError();
		//err=cudaDeviceSynchronize();
		err=cudaFree(kernel_gpu);
		err=cudaFree(buffer_t);
	}
	end = clock();
	double duration = (double)(end - start) / CLOCKS_PER_SEC;
	//printf("gb1d %.4lf\n", duration);
}

__global__ void ConvVertical_gpu(float* image, int width, int height, float* kernel, int ksize, float* buffer_t)
{
	int rows, cols, r, c, i, halfsize;
	float* pixels = image;

	rows = height;
	cols = width;

	halfsize = ksize / 2;
	if(rows + ksize >= 10000)return;

	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= cols)return;
	else c = cur;
	
	int index = c * 10000;
	float* buffer = buffer_t + index;
	//cudaMalloc((double**)&buffer, 10000 * sizeof(double));

	for (i = 0; i < halfsize; i++)
	{
		buffer[i] = pixels[c];
		//atomicExch(buffer+i, pixels[c]);
	}
	for (i = 0; i < rows; i++)
	{
		buffer[halfsize + i] = pixels[i * cols + c];
		//atomicExch(buffer + halfsize + i, pixels[i * cols + c]);
	}
	for (i = 0; i < halfsize; i++)
	{
		buffer[halfsize + rows + i] = pixels[(rows - 1) * cols + c];
		//atomicExch(buffer + halfsize + rows + i, pixels[(rows - 1) * cols + c]);
	}
	
	ConvBufferFast_gpu(buffer, kernel, rows, ksize);
	
	for (r = 0; r < rows; r++)
	{
		//pixels[r * cols + c] = buffer[r];
		atomicExch(pixels + r * cols + c, buffer[r]);
	}
	//delete[] buffer;
}

__device__ void ConvBufferFast_gpu(float* buffer, float* kernel, int rsize, int ksize)
{
	int i;
	float* bp, * kp, * endkp;
	float sum;

	for (i = 0; i < rsize; i++) {
		sum = 0.0;
		bp = &buffer[i];
		kp = &kernel[0];
		endkp = &kernel[ksize];
		float tmp = 0; //Kahan's Summation Formula
		while (kp < endkp) {
			float eps = 0;
			tmp -= (*bp++) * (*kp++);
			eps = sum - tmp;
			tmp = (eps - sum) + tmp;
			sum = eps;
		}

		buffer[i] = sum;
	}
}