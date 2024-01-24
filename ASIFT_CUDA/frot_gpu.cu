#include"frot_gpu.cuh"

#ifndef G_PI
#define G_PI 3.14159265358979323846
#endif

__host__ __device__ void bound_gpu(int x, int y, float ca, float sa, int* xmin, int* xmax, int* ymin, int* ymax);


float* frot_gpu(float* in, float* out, int nx, int ny, int* nx_out, int* ny_out, float a, float b, char* k_flag)
{
    clock_t start = clock(), end;

    float ca, sa, xtrans, ytrans;
    int xmin, xmax, ymin, ymax, sx, sy;


    ca = (float)cos((double)(a) * G_PI / 180.0);
    sa = (float)sin((double)(a) * G_PI / 180.0);

    /********** Compute new image location **********/
    if (k_flag)
    {
        /* crop image and fix center */
        xmin = ymin = 0;
        xmax = nx - 1;
        ymax = ny - 1;
        xtrans = 0.5f * ((float)(nx - 1) * (1.f - ca) + (float)(ny - 1) * sa);
        ytrans = 0.5f * ((float)(ny - 1) * (1.f - ca) - (float)(nx - 1) * sa);
    }
    else
    {
        /* extend image size to include the whole input image */
        xmin = xmax = ymin = ymax = 0;
        bound_gpu(nx - 1, 0, ca, sa, &xmin, &xmax, &ymin, &ymax);
        bound_gpu(0, ny - 1, ca, sa, &xmin, &xmax, &ymin, &ymax);
        bound_gpu(nx - 1, ny - 1, ca, sa, &xmin, &xmax, &ymin, &ymax);
        xtrans = ytrans = 0.0;
    }
    sx = xmax - xmin + 1;
    sy = ymax - ymin + 1;


    *nx_out = sx;
    *ny_out = sy;

    float* out_dev;
    int err_code=cudaMalloc((float**)&out_dev, sx * sy * sizeof(float));
    
    /********** Rotate image **********/
    int grid_size = sqrt(sx * sy) / 32 + 1;
    dim3 grid(grid_size, grid_size), block(32, 32);
    frot_MT << <grid, block >> > (in, out_dev, sx, sy, xmin, ymin, ca, sa, xtrans, ytrans, nx, ny, b, sx * sy);
    cudaError_t err = cudaGetLastError();
    //err = cudaDeviceSynchronize();

    end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    //cout << "frot " << duration << endl;

    return out_dev;
}

__global__ void frot_MT(float* in, float* out, int sx, int sy, int xmin, int ymin, float ca, float sa, float xtrans, float ytrans, int nx, int ny, float b, int total)
{
    int block_size = blockDim.x * blockDim.y;
    int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
    if (cur >= total)
        return;
    
    int x = cur / sy, y = cur % sy;
    x = x + xmin;
    y = y + ymin;

    int x1, y1, adr, tx1, ty1, tx2, ty2;
    float xp, yp, ux, uy, a11, a12, a21, a22;

    xp = ca * (float)x - sa * (float)y + xtrans;
    yp = sa * (float)x + ca * (float)y + ytrans;
    x1 = (int)floorf(xp);
    y1 = (int)floorf(yp);
    ux = xp - (float)x1;
    uy = yp - (float)y1;
    adr = y1 * nx + x1;
    tx1 = (x1 >= 0 && x1 < nx);
    tx2 = (x1 + 1 >= 0 && x1 + 1 < nx);
    ty1 = (y1 >= 0 && y1 < ny);
    ty2 = (y1 + 1 >= 0 && y1 + 1 < ny);

    a11 = (tx1 && ty1 ? in[adr] : b);
    a12 = (tx1 && ty2 ? in[adr + nx] : b);
    a21 = (tx2 && ty1 ? in[adr + 1] : b);
    a22 = (tx2 && ty2 ? in[adr + nx + 1] : b);

    //out[(y - ymin) * sx + x - xmin] = (1.f - uy) * ((1.f - ux) * a11 + ux * a21) + uy * ((1.f - ux) * a12 + ux * a22);
    atomicExch(out + (y - ymin) * sx + x - xmin, (1.f - uy) * ((1.f - ux) * a11 + ux * a21) + uy * ((1.f - ux) * a12 + ux * a22));
}

__host__ __device__ void bound_gpu(int x, int y, float ca, float sa, int* xmin, int* xmax, int* ymin, int* ymax)
{
    int rx, ry;

    rx = (int)floor(ca * (float)x + sa * (float)y);
    ry = (int)floor(-sa * (float)x + ca * (float)y);
    if (rx < *xmin) *xmin = rx; if (rx > *xmax) *xmax = rx;
    if (ry < *ymin) *ymin = ry; if (ry > *ymax) *ymax = ry;
}