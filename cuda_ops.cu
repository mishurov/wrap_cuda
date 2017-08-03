#include "cuda_ops.h"
#include <stdio.h>

// kernels
namespace kernels {

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
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

__global__ void compute_weights(float local,
								double * distances,
								unsigned int deformed_points_count,
								unsigned int triangles_count,
								double *weights,
								double *weights_sums) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= deformed_points_count || j >= triangles_count)
		return;
	double distance = distances[i * triangles_count + j];
	weights[i * triangles_count + j] = 1 / (1 + pow(distance, (double)local));
	weights_sums[i] = 0;
};

__global__ void sum_weights(unsigned int deformed_points_count,
							unsigned int triangles_count,
							double *weights,
							double *weights_sums) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= deformed_points_count || j >= triangles_count)
		return;
	atomicAdd(&(weights_sums[i]),
			  weights[i * triangles_count + j]);
};

__global__ void normalise_weights(unsigned int deformed_points_count,
								  unsigned int triangles_count,
								  double *weights,
								  double *weights_sums,
								  double *normalised_weights) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= deformed_points_count || j >= triangles_count)
		return;
	unsigned int offset = i * triangles_count + j;
	normalised_weights[offset] = weights[offset] / weights_sums[i];
};

__global__ void transform(double *in_vec, float *mat,
						  unsigned int num, double *out_vec) {
	int tid = (blockIdx.y * gridDim.x * blockDim.x) +
			  (blockIdx.x * blockDim.x) +
			  (threadIdx.x);
	
	int sub_id = threadIdx.x % 4;

	double sum = in_vec[tid - sub_id] * mat[sub_id];
	sum += in_vec[tid - sub_id + 1] * mat[4 + sub_id];
	sum += in_vec[tid - sub_id + 2] * mat[8 + sub_id];
	sum += in_vec[tid - sub_id + 3] * mat[12 + sub_id];
	out_vec[tid] = sum;
}

}

// wrapper
int CudaDeviceCount()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
};

void CudaComputeWeights(
	double *normalised_weights,
	double *contol_space_points,
	float local,
	double *distances,
	double *deformed_points,
	unsigned int deformed_points_count,
	unsigned int triangles_count,
	unsigned int *triangles_vertices,
	double *ref_vertices_cu,
	unsigned int ref_vertices_count,
	double *reference_matrices_cu
) {
	cudaError_t error;

	unsigned int grid_area = deformed_points_count * triangles_count;
	// internal vars
	double *d_weights_sums;
	error = cudaMalloc((void **)&d_weights_sums,
			   sizeof(double) * deformed_points_count);
	cudaMemset(d_weights_sums, 0, sizeof(double) * deformed_points_count);

	double *d_weights;
	error = cudaMalloc((void **)&d_weights, sizeof(double) * grid_area);
	cudaMemset(d_weights, 0, sizeof(double) * grid_area);

	// output vars
	double *d_normalised_weights;
	error = cudaMalloc((void **)&d_normalised_weights,
					   sizeof(double) * grid_area);
	cudaMemset(d_normalised_weights, 0, sizeof(double) * grid_area);

	// input vars
	double *d_distances;
	error = cudaMalloc((void **)&d_distances, sizeof(double) * grid_area);
	error = cudaMemcpy(d_distances, distances, sizeof(double) * grid_area,
			   		   cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(deformed_points_count / threadsPerBlock.x,
				   triangles_count / threadsPerBlock.y);

	kernels::compute_weights<<<numBlocks, threadsPerBlock>>>(
		local, d_distances, deformed_points_count, triangles_count, d_weights, d_weights_sums
	);
	error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("Compute weights: %s\n", cudaGetErrorString(error));
	
	//cudaDeviceSynchronize();
	kernels::sum_weights<<<numBlocks, threadsPerBlock>>>(
		deformed_points_count, triangles_count, d_weights, d_weights_sums
	);
	error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("Sum weights: %s\n", cudaGetErrorString(error));

	kernels::normalise_weights<<<numBlocks, threadsPerBlock>>>(
		deformed_points_count, triangles_count, d_weights,
		d_weights_sums, d_normalised_weights 
	);
	if(error != cudaSuccess)
		printf("Normalise weights: %s\n", cudaGetErrorString(error));

	error = cudaGetLastError();

	error = cudaMemcpy(normalised_weights, d_normalised_weights,
			   sizeof(double) * deformed_points_count * triangles_count,
			   cudaMemcpyDeviceToHost);
	if(error != cudaSuccess)
		printf("MemCpy DevToHost: %s\n", cudaGetErrorString(error));

	//cudaDeviceSynchronize();

	cudaFree(d_distances);
	cudaFree(d_weights);
	cudaFree(d_weights_sums);
	cudaFree(d_normalised_weights);
};
