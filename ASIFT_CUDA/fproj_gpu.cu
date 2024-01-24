#include "fproj_gpu.cuh"



float* fproj_gpu(float* in, float* out, int nx, int ny, int* sx, int* sy, float bg, int o, float p, char* i, float X1, float Y1, float X2, float Y2, float X3, float Y3, float* x4, float* y4)
{
	clock_t start=clock(), end;

	int    n1, n2, x, y, xi, yi, adr, dx, dy;
	float  res, xx = 0, yy = 0, xp, yp, ux, uy, a, b, fx, fy, x12, x13, y12, y13;
	float  cx[12], cy[12], ak[13];
	/* Fimage ref,coeffs; */
	float* ref, * coeffs;


	/* CHECK ORDER */
	if (o != 0 && o != 1 && o != -3 &&
		o != 3 && o != 5 && o != 7 && o != 9 && o != 11)
	{
		printf("unrecognized interpolation order.\n");
		exit(-1);
	}

	/* ALLOCATE NEW IMAGE */
	if (o >= 3) {
		cudaMalloc((float**)&coeffs, nx * ny * sizeof(float));
		double* buffer;
		cudaMalloc((double**)&buffer, 2 * nx * ny * sizeof(double));

		finvspline_gpu(in, o, coeffs, nx, ny,buffer);
		cudaError_t err = cudaGetLastError();
		//err=cudaDeviceSynchronize();
		cudaFree(buffer);

		end = clock();
		double duration = (double)(end - start) / CLOCKS_PER_SEC;
		//cout << "fproj partly use " << duration << " seconds." << endl;

		ref = coeffs;
		if (o > 3) init_splinen_gpu(ak, o);
	}
	else {
		ref = in;
	}

	/* COMPUTE NEW BASIS */
	if (i) {
		x12 = (X2 - X1) / (float)nx;
		y12 = (Y2 - Y1) / (float)nx;
		x13 = (X3 - X1) / (float)ny;
		y13 = (Y3 - Y1) / (float)ny;
	}
	else {
		x12 = (X2 - X1) / (float)(*sx);
		y12 = (Y2 - Y1) / (float)(*sx);
		x13 = (X3 - X1) / (float)(*sy);
		y13 = (Y3 - Y1) / (float)(*sy);
	}

	if (y4) {
		xx = ((*x4 - X1) * (Y3 - Y1) - (*y4 - Y1) * (X3 - X1)) / ((X2 - X1) * (Y3 - Y1) - (Y2 - Y1) * (X3 - X1));
		yy = ((*x4 - X1) * (Y2 - Y1) - (*y4 - Y1) * (X2 - X1)) / ((X3 - X1) * (Y2 - Y1) - (Y3 - Y1) * (X2 - X1));
		a = (yy - 1.f) / (1.f - xx - yy);
		b = (xx - 1.f) / (1.f - xx - yy);
	}
	else
	{
		a = b = 0.0;
	}

	float* out_dev;
	int err_code=cudaMalloc((float**)&out_dev, (*sx) * (*sy) * sizeof(float));

	int total = (*sx) * (*sy);
	int gsize = sqrt(total / 1024) + 1;
	dim3 grid(gsize, gsize), block(32, 32);
	char* i_dev;
	cudaMalloc((char**)&i_dev, sizeof(char));

	fproj_MT << <grid, block >> > (in, out_dev, ref, nx, ny, (*sx), (*sy), o, p, i_dev, bg, xx, yy, a, b, x12, x13, y12, y13, X1, Y1, total);
	//cudaError_t err = cudaDeviceSynchronize();
	cudaFree(coeffs);
	cudaFree(i_dev);
	/*
	cudaFree(CX);
	cudaFree(CY);
	cudaFree(AK);
	*/

	end = clock();
	double duration = (double)(end - start) / CLOCKS_PER_SEC;
	//cout << "fproj " << duration << endl;

	return out_dev;
}

__device__ float get_cxy(int index, float t)
{
	float tmp = 1.f - t;

	if (index == 0)return 0.1666666666f * t * t * t;
	else if (index == 1)return 0.6666666666f - 0.5f * tmp * tmp * (1.f + t);
	else if (index == 2)return 0.6666666666f - 0.5f * t * t * (2.f - t);
	else if (index == 3)return 0.1666666666f * tmp * tmp * tmp;
}

__global__ void fproj_MT(float* in, float* out, float* ref, int nx, int ny, int sx, int sy, int o, float p, char* i, float bg, float xx, float yy, float a, float b, float x12, float x13, float y12, float y13, float X1, float Y1, int total)
{
	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= total)
		return;
	int x = cur / sy, y = cur % sy;
	/*
	float* TEMP = temp + cur * 37;
	//memcpy(TEMP, package, 37 * sizeof(float));
	float* cx = TEMP, * cy = TEMP + 12, * ak = TEMP + 24;
	*/
	int n1, n2, xi, yi, adr;
	float  xp, yp, ux, uy, fx, fy, d;
	float res;
	/* COMPUTE LOCATION IN INPUT IMAGE */
	if (i) {
		xx = 0.5f + (((float)x - X1) * y13 - ((float)y - Y1) * x13) / (x12 * y13 - y12 * x13);
		yy = 0.5f - (((float)x - X1) * y12 - ((float)y - Y1) * x12) / (x12 * y13 - y12 * x13);
		d = 1.f - (a / (a + 1.f)) * xx / (float)nx - (b / (b + 1.f)) * yy / (float)ny;
		xp = xx / ((a + 1.f) * d);
		yp = yy / ((b + 1.f) * d);
	}
	else {
		fx = (float)x + 0.5f;
		fy = (float)y + 0.5f;
		d = a * fx / (float)(sx)+b * fy / (float)(sy)+1.f;
		xx = (a + 1.f) * fx / d;
		yy = (b + 1.f) * fy / d;
		xp = X1 + xx * x12 + yy * x13;
		yp = Y1 + xx * y12 + yy * y13;
	}


	/* INTERPOLATION */
	if (o == 0) {

		/* zero order interpolation (pixel replication) */
		xi = (int)floorf((double)xp);
		yi = (int)floorf((double)yp);
		/*	if (xi<0 || xi>=in->ncol || yi<0 || yi>=in->nrow)*/
		if (xi < 0 || xi >= nx || yi < 0 || yi >= ny)
			res = bg;
		else
			/* res = in->gray[yi*in->ncol+xi]; */
			res = in[yi * nx + xi];
	}
	else {

		/* higher order interpolations */
		if (xp<0. || xp>(float)nx || yp<0. || yp>(float)ny) res = bg;
		else {
			xp -= 0.5; yp -= 0.5;
			xi = (int)floorf((double)xp);
			yi = (int)floorf((double)yp);
			ux = xp - (float)xi;
			uy = yp - (float)yi;
			
			if (o == 1)
			{
				n2 = 1;
				//cx[0] = ux;	cx[1] = 1.f - ux;
				//cy[0] = uy; cy[1] = 1.f - uy;
			}
			else if (o == -1)
			{
				n2 = 2;
				//keys_gpu(cx, ux, p);
				//keys_gpu(cy, uy, p);
			}
			else if (o == 3)
			{
				n2 = 2;
				//spline3_gpu(cx, ux);
				//spline3_gpu(cy, uy);
			}
			else
			{
				n2 = (1 + o) / 2;
				//splinen_gpu(cx, ux, ak, o);
				//splinen_gpu(cy, uy, ak, o);
			}
			/*
			switch (o)
			{
			case 1: // first order interpolation (bilinear) 
				n2 = 1;
				cx[0] = ux;	cx[1] = 1.f - ux;
				cy[0] = uy; cy[1] = 1.f - uy;
				break;

			case -3: // third order interpolation (bicubic Keys' function) 
				n2 = 2;
				keys_gpu(cx, ux, p);
				keys_gpu(cy, uy, p);
				break;

			case 3: // spline of order 3 
				n2 = 2;
				spline3_gpu(cx, ux);
				spline3_gpu(cy, uy);
				break;

			default: // spline of order >3 
				n2 = (1 + o) / 2;
				splinen_gpu(cx, ux, ak, o);
				splinen_gpu(cy, uy, ak, o);
				break;
			}
			*/
			//printf("%d:3\n", cur);
			res = 0.;
			n1 = 1 - n2;
			/* this test saves computation time */
			if (xi + n1 >= 0 && xi + n2 < nx && yi + n1 >= 0 && yi + n2 < ny) {
				adr = yi * nx + xi;

				for (int i = n1; i <= n2; i++)
					for (int j = n1; j <= n2; j++)
						//res += cy[n2-dy]*cx[n2-dx]*ref->gray[adr+nx*dy+dx];
						res += get_cxy(n2 - i, uy) * get_cxy(n2 - j, ux) * ref[adr + nx * i + j];

			}
			else
				for (int i = n1; i <= n2; i++)
					for (int j = n1; j <= n2; j++)
						/*		res += cy[n2-dy]*cx[n2-dx]*v(ref,xi+dx,yi+dy,*bg); */
						res += get_cxy(n2 - i, uy) * get_cxy(n2 - j, ux) * v_gpu(ref, xi + j, yi + i, bg, nx, ny);
		}
	}
	atomicExch(out + y * sx + x, res);
	//printf("%d:4\n", cur);
	//out[y * (sx)+x] = res;
}



__device__ void keys_gpu(float* c, float t, float a)
{
	float t2, at;

	t2 = t * t;
	at = a * t;
	c[0] = a * t2 * (1.f - t);
	c[1] = (2.f * a + 3.f - (a + 2.f) * t) * t2 - at;
	c[2] = ((a + 2.f) * t - a - 3.f) * t2 + 1.f;
	c[3] = a * (t - 2.f) * t2 + at;
}

__device__ void spline3_gpu(float* c, float t)
{	
	float tmp = 1.f - t;

	c[0]= 0.1666666666f * t * t * t;
	c[1] = 0.6666666666f - 0.5f * tmp * tmp * (1.f + t);
	c[2] = 0.6666666666f - 0.5f * t * t * (2.f - t);
	c[3] = 0.1666666666f * tmp * tmp * tmp;
}

__device__ float ipow_gpu(float x, int n)
{
	float res;

	for (res = 1.; n; n >>= 1) {
		if (n & 1) res *= x;
		x *= x;
	}
	return(res);
}

__device__ void splinen_gpu(float* c, float t, float* a, int n)
{
	int i, k;
	float xn;

	memset((void*)c, 0, (n + 1) * sizeof(float));
	for (k = 0; k <= n + 1; k++) {
		xn = ipow_gpu(t + (float)k, n);
		for (i = k; i <= n; i++)
			c[i] += a[i - k] * xn;
	}
}

__device__ float v_gpu(float* in, int x, int y, float bg, int width, int height)
// float v(float *in, int x,int y,float bg, int width, int height)
{
	if (x < 0 || x >= width || y < 0 || y >= height)
		return(bg);
	else return(in[y * width + x]);
}

void init_splinen_gpu(float* a, int n)
{
	int k;

	a[0] = 1.;
	for (k = 2; k <= n; k++) a[0] /= (float)k;
	for (k = 1; k <= n + 1; k++)
		a[k] = -a[k - 1] * (float)(n + 2 - k) / (float)k;
}

__global__ void change_MT(float* in, double* out, int total)
{
	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= total)
		return;

	out[cur] = (double)in[cur];
}

__global__ void change2_MT(double* in, double* out, int nx, int ny)
{
	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= nx*ny)
		return;
	int x = cur / ny, y = cur % ny;

	out[x * ny + y] = in[y * nx + x];
}

__global__ void change3_MT(double* in, float* out, int nx, int ny)
{
	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= nx * ny)
		return;
	int x = cur / ny, y = cur % ny;

	out[y * nx + x] = (float)in[x * ny + y];
}

void finvspline_gpu(float* In, int order, float* Out, int width, int height,double* buffer)
// void finvspline(float *in,int order,float *out, int width, int height)
{
	float* in = In, * out = Out;
	double* c, * d, z[5];
	int npoles, nx, ny, x, y;

	ny = height; nx = width;

	/* initialize poles of associated z-filter */
	switch (order)
	{
	case 2: z[0] = -0.17157288;  /* sqrt(8)-3 */
		break;

	case 3: z[0] = -0.26794919;  /* sqrt(3)-2 */
		break;

	case 4: z[0] = -0.361341; z[1] = -0.0137254;
		break;

	case 5: z[0] = -0.430575; z[1] = -0.0430963;
		break;

	case 6: z[0] = -0.488295; z[1] = -0.0816793; z[2] = -0.00141415;
		break;

	case 7: z[0] = -0.53528; z[1] = -0.122555; z[2] = -0.00914869;
		break;

	case 8: z[0] = -0.574687; z[1] = -0.163035; z[2] = -0.0236323; z[3] = -0.000153821;
		break;

	case 9: z[0] = -0.607997; z[1] = -0.201751; z[2] = -0.0432226; z[3] = -0.00212131;
		break;

	case 10: z[0] = -0.636551; z[1] = -0.238183; z[2] = -0.065727; z[3] = -0.00752819;
		z[4] = -0.0000169828;
		break;

	case 11: z[0] = -0.661266; z[1] = -0.27218; z[2] = -0.0897596; z[3] = -0.0166696;
		z[4] = -0.000510558;
		break;

	default:
		printf("finvspline: order should be in 2..11.\n");
		return;
	}
	double* Z;
	cudaMalloc((double**)&Z, 5 * sizeof(double));
	cudaMemcpy(Z, z, 5 * sizeof(double), cudaMemcpyHostToDevice);
	npoles = order / 2;

	/* initialize double array containing image */
	c = buffer;
	d = buffer + nx * ny;

	dim3 grid, block(32, 32);

	int gsize = sqrt(nx * ny) / 32 + 1;
	grid = dim3(gsize, gsize);

	change_MT << <grid, block >> > (in, c, nx * ny);
	
	/*
	for (x = nx * ny; x--;)
	{
		c[x] = (double)in[x];
		if(x==nx*ny-1)printf("x:%d\n", x);
	}
	*/
	/* apply filter on lines */
	if (ny > 1024)gsize = sqrt(ny) / 32 + 1;
	else gsize = 1;
	grid = dim3(gsize, gsize);
	invspline1D_gpu << <grid, block >> > (c, nx, Z, npoles, ny);
	/*
	for (y = 0; y < ny; y++)
		invspline1D_gpu(c + y * nx, nx, z, npoles,ny);
	*/
	/* transpose */
	if (nx * ny > 1024)gsize = sqrt(nx * ny) / 32 + 1;
	else gsize = 1;
	grid = dim3(gsize, gsize);
	change2_MT << <grid, block >> > (c, d, nx, ny);
	/*
	for (x = 0; x < nx; x++)
		for (y = 0; y < ny; y++)
			d[x * ny + y] = c[y * nx + x];
	*/
	/* apply filter on columns */
	if (nx > 1024)gsize = sqrtf(nx) / 32 + 1;
	else gsize = 1;
	grid = dim3(gsize, gsize);
	invspline1D_gpu << <grid, block >> > (d, ny, Z, npoles, nx);
	/*
	for (x = 0; x < nx; x++)
		invspline1D_gpu(d + x * ny, ny, z, npoles);
	*/
	/* transpose directy into image */
	if (nx * ny > 1024)gsize = sqrtf(nx * ny) / 32 + 1;
	else gsize = 1;
	grid = dim3(gsize, gsize);
	change3_MT << <grid, block >> > (d, out, nx, ny);
	/*
	for (x = 0; x < nx; x++)
		for (y = 0; y < ny; y++)
		{
			//out[y * nx + x] = (float)(d[x * ny + y]);
			atomicExch(out + y * nx + x, (float)(d[x * ny + y]));
		}
	*/
	In = in;
	Out = out;
	/* free array */
	//free(d);
	//free(c);
}

__global__ void invspline1D_gpu(double* c, int size, double* z, int npoles, int total)
{
	int block_size = blockDim.x * blockDim.y;
	int cur = (blockIdx.y * gridDim.x + blockIdx.x) * block_size + threadIdx.y * blockDim.x + threadIdx.x;
	if (cur >= total)
		return;
	double* C = c + cur * size;
	
	double lambda;
	int n, k;

	/* normalization */
	for (k = npoles, lambda = 1.; k--;)
		lambda *= (1. - z[k]) * (1. - 1. / z[k]);
	
	for (n = size; n--;) C[n] *= lambda;
	
	/*----- Loop on poles -----*/
	for (k = 0; k < npoles; k++) {

		/* forward recursion */
		C[0] = initcausal_gpu(C, size, z[k]);
		for (n = 1; n < size; n++)
			C[n] += z[k] * C[n - 1];

		/* backwards recursion */
		C[size - 1] = initanticausal_gpu(C, size, z[k]);
		for (n = size - 1; n--;)
			C[n] = z[k] * (C[n + 1] - C[n]);

	}
}

__device__ double initcausal_gpu(double* c, int n, double z)
{
	double zk, z2k, iz, sum;
	int k;

	zk = z; iz = 1. / z;
	z2k = powf(z, (double)n - 1.);
	sum = c[0] + z2k * c[n - 1];
	z2k = z2k * z2k * iz;
	double t1 = 0;
	for (k = 1; k <= n - 2; k++) {
		double eps = 0;
		t1 -= ((zk + z2k) * c[k]);
		eps = sum - t1;
		t1 = (eps - sum) + t1;
		sum = eps;
		zk *= z;
		z2k *= iz;
	}
	return (sum / (1. - zk * zk));
}

__device__ double initanticausal_gpu(double* c, int n, double z)
{
	return((z / (z * z - 1.)) * (z * c[n - 2] + c[n - 1]));
}
