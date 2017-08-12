#include "cuda_ops.h"
#include <stdio.h>
#include <cublas_v2.h>

// kernels
namespace kernels {

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__device__ double3 sub_double3(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double3 cross_double3(double3 a, double3 b)
{
	return make_double3(a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

__device__ double dot_double3(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;;
}

__device__ double inv_len_double3(double3 v)
{
	return rsqrt(dot_double3(v, v));
}

__device__ double3 mult_double3_double(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}


__global__ void computeWeights(float local, double *D, double *W,
				unsigned int numPoints,
				unsigned int numElems)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numPoints || j >= numElems)
		return;

	unsigned int k = i * numElems + j;

	double d = D[k];
	W[k] = 1 / (1 + pow(d, (double)local));
};

__global__ void sumWeights(double *W, double *Wsum, unsigned int numPoints,
			unsigned int numElems)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numPoints || j >= numElems)
		return;

	unsigned int k = i * numElems + j;

	atomicAdd(&(Wsum[i]), W[k]);
};

__global__ void normaliseWeights(double *W, double *Wsum, double *Wnorm,
				unsigned int numPoints, unsigned int numElems)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numPoints || j >= numElems)
		return;

	unsigned int k = i * numElems + j;
	Wnorm[k] = W[k] / Wsum[i];
};

__global__ void computeTransformMats(double *R, unsigned int *Ridx,
				double **M, unsigned int numElems)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numElems)
		return;

	unsigned int a = Ridx[i * 3] * 3;
	unsigned int c = Ridx[i * 3 + 1] * 3;
	unsigned int b = Ridx[i * 3 + 2] * 3;
		
	double3 A = make_double3(R[a], R[a + 1], R[a + 2]);
	double3 B = make_double3(R[b], R[b + 1], R[b + 2]);
	double3 C = make_double3(R[c], R[c + 1], R[c + 2]);

	double3 E1 = sub_double3(C, A);
	double3 E2 = sub_double3(B, A);
	double3 E1xE2 = cross_double3(E1, E2);
	double inv_len = inv_len_double3(E1xE2);

	double3 E3 = mult_double3_double(E1xE2, inv_len);

	// store the matrices in the row-major order
	double mat[16] = {
		E1.x, E2.x, E3.x, 0,
		E1.y, E2.y, E3.y, 0,
		E1.z, E2.z, E3.z, 0,
		0,    0,    0,    1
	};

	for (int j = 0; j < 16; j++)
		M[i][j] = mat[j];
};

__global__ void computeControlPoints(double *P, double **M, double *C,
				unsigned int numPoints, unsigned int numElems)
{

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= numPoints || j >= numElems)
		return;

	double3 pp = make_double3(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]);

	double x = (
		M[j][0] * pp.x +
		M[j][1] * pp.y +
		M[j][2] * pp.z +
		M[j][3]
	);
	double y = (
		M[j][4] * pp.x +
		M[j][5] * pp.y +
		M[j][6] * pp.z +
		M[j][7]
	);
	double z = (
		M[j][8] * pp.x +
		M[j][9] * pp.y +
		M[j][10] * pp.z +
		M[j][11]
	);

	unsigned int k = i * numElems * 3 + j * 3;

	C[k] = x;
	C[k + 1] = y;
	C[k + 2] = z;
}


__global__ void transformVerts(double **M, double *C, double *Wnorm, double *P,
				unsigned int numPoints, unsigned int numElems)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= numPoints || j >= numElems)
		return;

	double w = Wnorm[i * numElems + j];

	unsigned int k = i * numElems * 3 + j * 3;
	double3 cc = make_double3(C[k], C[k + 1], C[k + 2]);

	double x = (
		M[j][0] * cc.x +
		M[j][1] * cc.y +
		M[j][2] * cc.z
	);
	double y = (
		M[j][4] * cc.x +
		M[j][5] * cc.y +
		M[j][6] * cc.z
	);
	double z = (
		M[j][8] * cc.x +
		M[j][9] * cc.y +
		M[j][10] * cc.z
	);

	double3 Pw = make_double3(x * w, y * w, z * w);

	atomicAdd(&(P[i * 3]), Pw.x);
	atomicAdd(&(P[i * 3 + 1]), Pw.y);
	atomicAdd(&(P[i * 3 + 2]), Pw.z);
};

}


int CudaDeviceCount()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
};


void CudaComputeControlPoints(double *h_P, unsigned int *h_Ridx, double *h_R,
			double *h_C, unsigned int numRefs,
			unsigned int numPoints, unsigned int numElems)
{
	cudaError_t error;
	cublasHandle_t handle;
	cublasStatus_t status;
	status = cublasCreate_v2(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		printf("CCP, cublas create: %d\n", status);

	int *d_info;
	cudaMalloc(&d_info, numElems * sizeof(int));

	unsigned int gridArea = numPoints * numElems;

	// matrices
	double **h_M = (double **)malloc(numElems * sizeof(double *));
	double **d_M, *d_M_flat;

	error = cudaMalloc(&d_M, numElems * sizeof(double *));
	if (error != cudaSuccess)
		printf("CCP, matrices alloc: %s\n", cudaGetErrorString(error));

	cudaMalloc(&d_M_flat, sizeof(double) * numElems * 16);
	h_M[0] = d_M_flat;
	for (int i = 1; i < numElems; i++)
		h_M[i] = h_M[i - 1] + (4 * 4);
	cudaMemcpy(d_M, h_M, numElems * sizeof(double *),
		cudaMemcpyHostToDevice);

	// indices
	unsigned int *d_Ridx;
	error = cudaMalloc((void **)&d_Ridx,
			sizeof(unsigned int) * numElems * 3);
	if (error != cudaSuccess)
		printf("CCP, indices alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_Ridx, h_Ridx, sizeof(unsigned int) * numElems * 3,
		cudaMemcpyHostToDevice);

	// reference vertices
	double *d_R;
	error = cudaMalloc((void **)&d_R, sizeof(double) * numRefs * 3);
	if (error != cudaSuccess)
		printf("CCP, reference alloc: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_R, h_R, sizeof(double) * numRefs * 3,
			cudaMemcpyHostToDevice);

	// inverse matrices
	double **h_Minv = (double **)malloc(numElems * sizeof(double *));
	double **d_Minv, *d_Minv_flat;

	error = cudaMalloc(&d_Minv, numElems * sizeof(double *));
	if (error != cudaSuccess)
		printf("CCP, inv matrices alloc: %s\n", cudaGetErrorString(error));

	cudaMalloc(&d_Minv_flat, sizeof(double) * numElems * 16);
	h_Minv[0] = d_Minv_flat;
	for (int i = 1; i < numElems; i++)
		h_Minv[i] = h_Minv[i - 1] + (4 * 4);
	cudaMemcpy(d_Minv, h_Minv, numElems * sizeof(double *),
		cudaMemcpyHostToDevice);

	// points
	double *d_P;
	error = cudaMalloc((void **)&d_P, sizeof(double) * numPoints * 3);
	if(error != cudaSuccess)
		printf("CCP, inv points alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_P, h_P, sizeof(double) * numPoints * 3,
					   cudaMemcpyHostToDevice);

	// control points
	double *d_C;
	error = cudaMalloc((void **)&d_C, sizeof(double) * gridArea * 3);
	cudaMemset(d_C, 0, sizeof(double) * gridArea * 3);

	// computations
	dim3 threadsMatPerBlock(64, 1);
	dim3 numMatBlocks(numElems / threadsMatPerBlock.x + 1, 1);

	kernels::computeTransformMats<<<numMatBlocks, threadsMatPerBlock>>>(
		d_R, d_Ridx, d_M, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CCP, compute matrices: %s\n", cudaGetErrorString(error));

	cublasDmatinvBatched(handle, 4, (const double**)d_M,
					4, d_Minv, 4, d_info, numElems);

	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CCP, invert matrices: %s\n", cudaGetErrorString(error));

	cudaDeviceSynchronize();

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(numPoints / threadsPerBlock.x + 1,
			numElems / threadsPerBlock.y + 1);

	kernels::computeControlPoints<<<numBlocks, threadsPerBlock>>>(
		d_P, d_Minv, d_C, numPoints, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CCP, compute controls: %s\n", cudaGetErrorString(error));


	error = cudaMemcpy(h_C, d_C, sizeof(double) * gridArea * 3,
				cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("CCP, dev to host: %s\n", cudaGetErrorString(error));

	cudaFree(d_M);
	cudaFree(d_M_flat);
	cudaFree(d_R);
	cudaFree(d_Ridx);
	cudaFree(d_Minv);
	cudaFree(d_Minv_flat);
	cudaFree(d_P);
	cudaFree(d_C);
	free(h_M);
	free(h_Minv);
}


void CudaComputeWeights(float local, double *h_D, double *h_Wnorm,
			unsigned int numPoints, unsigned int numElems)
{
	cudaError_t error;
	unsigned int gridArea = numPoints * numElems;

	double *d_Wsum;
	error = cudaMalloc((void **)&d_Wsum, sizeof(double) * numPoints);
	if (error != cudaSuccess)
		printf("CW, sums alloc: %s\n", cudaGetErrorString(error));
	cudaMemset(d_Wsum, 0, sizeof(double) * numPoints);

	double *d_W;
	error = cudaMalloc((void **)&d_W, sizeof(double) * gridArea);
	if (error != cudaSuccess)
		printf("CW, weights alloc: %s\n", cudaGetErrorString(error));
	cudaMemset(d_W, 0, sizeof(double) * gridArea);

	double *d_Wnorm;
	error = cudaMalloc((void **)&d_Wnorm, sizeof(double) * gridArea);
	if (error != cudaSuccess)
		printf("CW, norm alloc: %s\n", cudaGetErrorString(error));
	cudaMemset(d_Wnorm, 0, sizeof(double) * gridArea);

	double *d_D;
	error = cudaMalloc((void **)&d_D, sizeof(double) * gridArea);
	if (error != cudaSuccess)
		printf("CW, dist alloc: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_D, h_D, sizeof(double) * gridArea,
			cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(numPoints / threadsPerBlock.x + 1,
			numElems / threadsPerBlock.y + 1);
	
	kernels::computeWeights<<<numBlocks, threadsPerBlock>>>(
		local, d_D, d_W, numPoints, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CW, compute weights: %s\n", cudaGetErrorString(error));

	kernels::sumWeights<<<numBlocks, threadsPerBlock>>>(
		d_W, d_Wsum, numPoints, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CW, sum weights: %s\n", cudaGetErrorString(error));

	kernels::normaliseWeights<<<numBlocks, threadsPerBlock>>>(
		d_W, d_Wsum, d_Wnorm, numPoints, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("CW, norm weights: %s\n", cudaGetErrorString(error));

	error = cudaMemcpy(h_Wnorm, d_Wnorm, sizeof(double) * gridArea,
			cudaMemcpyDeviceToHost);

	if(error != cudaSuccess)
		printf("CW, dev to host: %s\n", cudaGetErrorString(error));

	cudaFree(d_Wsum);
	cudaFree(d_W);
	cudaFree(d_Wnorm);
	cudaFree(d_D);
};


void CudaApplyDeform(double *h_C, double *h_Wnorm, unsigned int *h_DRidx,
		double *h_DR, double *h_P, unsigned int numDriv,
		unsigned int numPoints, unsigned int numElems)
{
	cudaError_t error;
	unsigned int gridArea = numPoints * numElems;
	
	double *d_P;
	error = cudaMalloc((void **)&d_P, sizeof(double) * numPoints * 3);
	if (error != cudaSuccess)
		printf("D, points alloc: %s\n", cudaGetErrorString(error));

	cudaMemset(d_P, 0, sizeof(double) * numPoints * 3);

	double **h_M = (double **)malloc(numElems * sizeof(double *));
	double **d_M, *d_M_flat;

	error = cudaMalloc(&d_M, numElems * sizeof(double *));
	if (error != cudaSuccess)
		printf("D, matrices alloc: %s\n", cudaGetErrorString(error));

	cudaMalloc(&d_M_flat, sizeof(double) * numElems * 16);
	h_M[0] = d_M_flat;
	for (int i = 1; i < numElems; i++)
		h_M[i] = h_M[i - 1] + (4 * 4);
	cudaMemcpy(d_M, h_M, numElems * sizeof(double *),
		cudaMemcpyHostToDevice);

	double *d_C;
	error = cudaMalloc((void **)&d_C, sizeof(double) * gridArea * 3);
	if (error != cudaSuccess)
		printf("D, controls alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_C, h_C, sizeof(double) * gridArea * 3,
		cudaMemcpyHostToDevice);
	
	double *d_Wnorm;
	error = cudaMalloc((void **)&d_Wnorm, sizeof(double) * gridArea);
	if (error != cudaSuccess)
		printf("D, norm alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_Wnorm, h_Wnorm, sizeof(double) * gridArea,
		cudaMemcpyHostToDevice);


	unsigned int *d_DRidx;
	error = cudaMalloc((void **)&d_DRidx, sizeof(unsigned int) * numElems * 3);
	if (error != cudaSuccess)
		printf("D, indices alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_DRidx, h_DRidx, sizeof(unsigned int) * numElems * 3,
		cudaMemcpyHostToDevice);

	double *d_DR;
	error = cudaMalloc((void **)&d_DR, sizeof(double) * numDriv * 3);
	if (error != cudaSuccess)
		printf("D, vertices alloc: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_DR, h_DR, sizeof(double) * numDriv * 3,
		cudaMemcpyHostToDevice);


	dim3 threadsMatPerBlock(64, 1);
	dim3 numMatBlocks(numElems / threadsMatPerBlock.x + 1, 1);

	kernels::computeTransformMats<<<numMatBlocks, threadsMatPerBlock>>>(
		d_DR, d_DRidx, d_M, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("D, compute matrices: %s\n", cudaGetErrorString(error));


	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(numPoints / threadsPerBlock.x + 1,
			numElems / threadsPerBlock.y + 1);

	kernels::transformVerts<<<numBlocks, threadsPerBlock>>>(
		d_M, d_C, d_Wnorm, d_P, numPoints, numElems
	);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("D, transform: %s\n", cudaGetErrorString(error));

	error = cudaMemcpy(h_P, d_P, sizeof(double) * numPoints * 3,
			cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
		printf("D, dev to host: %s\n", cudaGetErrorString(error));

	cudaFree(d_P);
	cudaFree(d_M);
	cudaFree(d_M_flat);
	cudaFree(d_C);
	cudaFree(d_Wnorm);
	cudaFree(d_DRidx);
	cudaFree(d_DR);
	free(h_M);
};
