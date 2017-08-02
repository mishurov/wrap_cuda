#ifndef MAYA_CUDA_OPS_H_
#define MAYA_CUDA_OPS_H_

int CudaDeviceCount();

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
);

#endif  // MAYA_CUDA_OPS_H_
