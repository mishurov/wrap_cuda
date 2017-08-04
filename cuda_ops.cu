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

inline __device__ double3 double3_sub(double3 a, double3 b) {
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ double3 double3_cross(double3 a, double3 b) {
	return make_double3(a.y * b.z - a.z * b.y,
						a.z * b.x - a.x * b.z,
						a.x * b.y - a.y * b.x);
}

inline __device__ double double3_length(double3 a) {
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline __device__ double3 double3_double_div(double3 a, double b) {
	if (b != 0)
    	return make_double3(a.x / b, a.y / b, a.z / b);
	else
    	return make_double3(1, 1, 1);
}

__global__ void create_def_matrices(
			unsigned int triangles_count,
			unsigned int *triangles_indices,
			double *vertices,
			double *res
) {
	unsigned int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tri_idx >= triangles_count)
		return;
	unsigned int index_A = triangles_indices[tri_idx * 3];
	unsigned int index_C = triangles_indices[tri_idx * 3 + 1];
	unsigned int index_B = triangles_indices[tri_idx * 3 + 2];
		
	double3 A = make_double3(
		vertices[index_A],
		vertices[index_A + 1],
		vertices[index_A + 2]
	);
	double3 B = make_double3(
		vertices[index_B],
		vertices[index_B + 1],
		vertices[index_B + 2]
	);
	double3 C = make_double3(
		vertices[index_C],
		vertices[index_C + 1],
		vertices[index_C + 2]
	);
	double3 E1 = double3_sub(C, A);
	double3 E2 = double3_sub(B, A);
	double3 E1_E2_cross = double3_cross(E1, E2);
	double3 E3 = double3_double_div(E1_E2_cross, double3_length(E1_E2_cross));

	/*
	E1.x E1.y E1.z,
	E2.x E2.y E2.z,
	E3.x E3.y E3.z
	*/
	unsigned int mat_idx = tri_idx * 9;
	res[mat_idx] = E1.x;
	res[mat_idx + 1] = E1.y;
	res[mat_idx + 2] = E1.z;
	res[mat_idx + 3] = E2.x;
	res[mat_idx + 4] = E2.y;
	res[mat_idx + 5] = E2.z;
	res[mat_idx + 6] = E3.x;
	res[mat_idx + 7] = E3.y;
	res[mat_idx + 8] = E3.z;
};

__device__ double3 double3_double_mult(double3 a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__device__ double3 double3_add(double3 a, double3 b) {
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void apply_deform(unsigned int deformed_points_count,
							 unsigned int triangles_count,
							 double *deformed_points,
							 double *mats,
							 double *cs_pts,
							 double *normalised_weights,
							 double *res_points)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= deformed_points_count || j >= triangles_count)
		return;
	unsigned int offset = i * triangles_count + j;

	double3 point = make_double3(
		deformed_points[i],
		deformed_points[i + 1],
		deformed_points[i + 2]
	);
	double weight = normalised_weights[offset];
	double3 cs_point = make_double3(
		cs_pts[offset],
		cs_pts[offset + 1],
		cs_pts[offset + 2]
	);

	// multiply by matrix
	unsigned int idx = j * 9;

	double x = (
		mats[idx] * cs_point.x +
		mats[idx + 1] * cs_point.y +
		mats[idx + 2] * cs_point.z
	);
	double y = (
		mats[idx + 3] * cs_point.x +
		mats[idx + 4] * cs_point.y +
		mats[idx + 5] * cs_point.z
	);
	double z = (
		mats[idx + 6] * cs_point.x +
		mats[idx + 7] * cs_point.y +
		mats[idx + 8] * cs_point.z
	);
	double3 control_point = make_double3(x, y, z);


	control_point = double3_double_mult(control_point, weight);
	atomicAdd(&(res_points[i]), control_point.x);
	atomicAdd(&(res_points[i + 1]), control_point.y);
	atomicAdd(&(res_points[i + 2]), control_point.z);
};

}

// wrappers
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
	dim3 numBlocks(deformed_points_count / threadsPerBlock.x + 1,
				   triangles_count / threadsPerBlock.y + 1);

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

	cudaFree(d_distances);
	cudaFree(d_weights);
	cudaFree(d_weights_sums);
	cudaFree(d_normalised_weights);
};


void CudaApplyDeform(
	double *out_points,
	double *cs_points,
	double *points,
	unsigned int deformed_points_count,
	unsigned int *triangles_indices,
	unsigned int triangles_count,
	double *driver_vertices_cu,
	unsigned int driver_vertices_count,
	double *normalised_weights,
	double *mats
) {
	cudaError_t error;
	unsigned int grid_area = deformed_points_count * triangles_count;
	
	// out vars
	double *d_out_points;
	error = cudaMalloc((void **)&d_out_points,
			   sizeof(double) * deformed_points_count * 3);
	cudaMemset(d_out_points, 0, sizeof(double) * deformed_points_count * 3);
	if(error != cudaSuccess)
		printf("outs: %s\n", cudaGetErrorString(error));
	
	// internal vars
	double *d_mats;
	error = cudaMalloc((void **)&d_mats,
			   sizeof(double) * triangles_count * 9);
	cudaMemset(d_mats, 0, sizeof(double) * triangles_count * 9);
	/*
	if(error != cudaSuccess)
		printf("mats: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_mats, mats,
						sizeof(double) * triangles_count * 9,
					   cudaMemcpyHostToDevice);
	*/

	// in vars
	double *d_cs_points;
	error = cudaMalloc((void **)&d_cs_points, sizeof(double) * grid_area * 3);
	error = cudaMemcpy(d_cs_points, cs_points, sizeof(double) * grid_area * 3,
					   cudaMemcpyHostToDevice);
	double *d_points;
	error = cudaMalloc((void **)&d_points,
					   sizeof(double) * deformed_points_count * 3);
	error = cudaMemcpy(d_points, points,
					   sizeof(double) * deformed_points_count * 3,
					   cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
		printf("cpy pts: %s\n", cudaGetErrorString(error));

	unsigned int *d_triangles_indices;
	error = cudaMalloc((void **)&d_triangles_indices,
					   sizeof(unsigned int) * triangles_count * 3);
	error = cudaMemcpy(d_triangles_indices, triangles_indices,
					   sizeof(unsigned int) * triangles_count * 3,
					   cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
		printf("cpy indices: %s\n", cudaGetErrorString(error));

	double *d_driver_vertices_cu;
	error = cudaMalloc((void **)&d_driver_vertices_cu,
					   sizeof(double) * driver_vertices_count * 3);
	error = cudaMemcpy(d_driver_vertices_cu, driver_vertices_cu,
					   sizeof(double) * driver_vertices_count * 3,
					   cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
		printf("cpy drv verts: %s\n", cudaGetErrorString(error));

	double *d_normalised_weights;
	error = cudaMalloc((void **)&d_normalised_weights,
					   sizeof(double) * grid_area);
	error = cudaMemcpy(d_normalised_weights, normalised_weights,
					   sizeof(double) * grid_area,
					   cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
		printf("cpy norm weights: %s\n", cudaGetErrorString(error));
	
	dim3 threadsMatPerBlock(64, 1);
	dim3 numMatBlocks(triangles_count / threadsMatPerBlock.x + 1, 1);

	kernels::create_def_matrices<<<numMatBlocks, threadsMatPerBlock>>>(
		triangles_count,
		d_triangles_indices,
		d_driver_vertices_cu,
		d_mats
	);
	error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("Create def mats: %s\n", cudaGetErrorString(error));

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(deformed_points_count / threadsPerBlock.x,
				   triangles_count / threadsPerBlock.y);

	kernels::apply_deform<<<numBlocks, threadsPerBlock>>>(
		deformed_points_count,
		triangles_count,
		d_points,
		d_mats,
		d_cs_points,
		d_normalised_weights,
		d_out_points
	);
	error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("Apply deform: %s\n", cudaGetErrorString(error));

	error = cudaMemcpy(out_points, d_out_points,
			   sizeof(double) * deformed_points_count * 3,
			   cudaMemcpyDeviceToHost);
	
	if(error != cudaSuccess)
		printf("Deform DevToHost: %s\n", cudaGetErrorString(error));
	/*
	error = cudaMemcpy(mats, d_mats,
			   sizeof(double) * triangles_count * 9,
			   cudaMemcpyDeviceToHost);
	*/

	cudaFree(d_out_points);
	cudaFree(d_mats);
	cudaFree(d_cs_points);
	cudaFree(d_points);
	cudaFree(d_triangles_indices);
	cudaFree(d_driver_vertices_cu);
	cudaFree(d_normalised_weights);
};
