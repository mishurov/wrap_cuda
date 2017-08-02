#include "cuda_ops.h"

// kernels
namespace kernels {

__global__ void compute_weights(float local,
								double * deformed_points,
								unsigned int deformed_points_count,
								unsigned int * triangles_vertices,
								double * ref_vertices_cu,
								double * reference_matrices_cu,
								double *contol_space_points,
								double *weights,
								double *weights_sums) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	double3 point = make_double3(
		deformed_points[i * 3],
		deformed_points[i * 3 + 1],
		deformed_points[i * 3 + 2]
	);
	unsigned int index_A = triangles_vertices[j * 3];
	unsigned int index_C = triangles_vertices[j * 3 + 1];
	unsigned int index_B = triangles_vertices[j * 3 + 2];
	double3 triangle[3];
	triangle[0] = make_double3(
		ref_vertices_cu[index_A * 3],
		ref_vertices_cu[index_A * 3 + 1],
		ref_vertices_cu[index_A * 3 + 2]
	);
	triangle[1] = make_double3(
		ref_vertices_cu[index_B * 3],
		ref_vertices_cu[index_B * 3 + 1],
		ref_vertices_cu[index_B * 3 + 2]
	);
	triangle[2] = make_double3(
		ref_vertices_cu[index_C * 3],
		ref_vertices_cu[index_C * 3 + 1],
		ref_vertices_cu[index_C * 3 + 2]
	);
	// todo func
	double distance = 1;
	weights[i * deformed_points_count + j] = 1 / (1 + pow(distance, local));
	// atomic add
	weights_sums[i] += weights[i * deformed_points_count + j];

	// matrix mult
	// contol_space_point = point * reference_matrices[j].inverse();
	// contol_space_points[j] = contol_space_point;
};

}

// wrapper
int CudaDeviceCount()
{
	int deviceCount = 0;
	cudaError_t cudaErrorId;
	cudaErrorId = cudaGetDeviceCount(&deviceCount);
	return deviceCount;
};

void CudaComputeWeights(
	double *normalized_weights,
	double *contol_space_points,
	float local,
	double *deformed_points,
	unsigned int deformed_points_count,
	unsigned int triangles_count,
	unsigned int *triangles_vertices,
	double *ref_vertices_cu,
	unsigned int ref_vertices_count,
	double *reference_matrices_cu
) {
	// output vars
	double * d_weights = NULL;
	cudaMalloc((void **)&d_weights,
			   sizeof(double) * deformed_points_count * triangles_count);
	double * d_weights_sums = NULL;
	cudaMalloc((void **)&d_weights_sums,
			   sizeof(double) * deformed_points_count);

	double * d_normalized_weights = NULL;
	cudaMalloc((void **)&d_normalized_weights,
			   sizeof(double) * deformed_points_count * triangles_count);
	double * d_contol_space_points = NULL;
	cudaMalloc((void **)&d_contol_space_points,
			   sizeof(double) * deformed_points_count * triangles_count * 3);
	// input vars
	double * d_deformed_points = NULL;
	cudaMalloc((void **)&d_deformed_points,
			   sizeof(double) * deformed_points_count * 3);
	cudaMemcpy(d_deformed_points, deformed_points,
			   sizeof(double) * deformed_points_count * 3,
			   cudaMemcpyHostToDevice);

	unsigned int *d_triangles_vertices = NULL;
	cudaMalloc((void **)&d_triangles_vertices,
			   sizeof(unsigned int) * triangles_count * 3);
	cudaMemcpy(d_triangles_vertices, triangles_vertices,
			   sizeof(unsigned int) * triangles_count * 3,
			   cudaMemcpyHostToDevice);

	double * d_ref_vertices_cu = NULL;
	cudaMalloc((void **)&d_ref_vertices_cu,
			   sizeof(double) * ref_vertices_count);
	cudaMemcpy(d_ref_vertices_cu, ref_vertices_cu,
			   sizeof(double) * ref_vertices_count,
			   cudaMemcpyHostToDevice);

	double * d_reference_matrices_cu = NULL;
	cudaMalloc((void **)&d_reference_matrices_cu,
			   sizeof(double) * triangles_count * 3 * 3);
	cudaMemcpy(d_reference_matrices_cu, ref_vertices_cu,
			   sizeof(double) * triangles_count * 3 * 3,
			   cudaMemcpyHostToDevice);

	// incorrect block/thread size
	dim3 threadsPerBlock(deformed_points_count, triangles_count);
	kernels::compute_weights<<<1, threadsPerBlock>>>(
		local,
		d_deformed_points,
		deformed_points_count,
		d_triangles_vertices,
		d_ref_vertices_cu,
		d_reference_matrices_cu,
		d_contol_space_points,
		d_weights,
		d_weights_sums
	);

};
