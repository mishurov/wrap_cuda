#ifndef MAYA_CUDA_OPS_H_
#define MAYA_CUDA_OPS_H_

int CudaDeviceCount();

void CudaComputeWeights(
	double *normalized_weights,
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
);

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
);

#endif  // MAYA_CUDA_OPS_H_
