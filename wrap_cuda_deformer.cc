/* ************************************************************************
 * Copyright 2013 Alexander Mishurov
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#include "wrap_cuda_deformer.h"
#include <stdio.h>

MTypeId WrapCudaDeformer::id(kWrapCudaDeformerID);
MObject WrapCudaDeformer::reference_surface_;
MObject WrapCudaDeformer::driver_surface_;
MObject WrapCudaDeformer::local_;
MObject WrapCudaDeformer::cuda_;

WrapCudaDeformer::WrapCudaDeformer() {}

WrapCudaDeformer::~WrapCudaDeformer() {
	MMessage::removeCallbacks(callback_ids_);
	callback_ids_.clear();
}

void WrapCudaDeformer::postConstructor()
{
	MStatus status;
	MObject o_node = thisMObject();
	MCallbackId callback_id;
	callback_id = MNodeMessage::addNodeDirtyPlugCallback(o_node,
						 	registrationCallback,
						 	this,
						 	&status);
	CHECK_MSTATUS(status);
	callback_ids_.append(callback_id);
	registration_phase_ = true;
}

void WrapCudaDeformer::registrationCallback(MObject& node,
					MPlug& plug,
					void* client_data)
{
	if (plug == cuda_ || plug == reference_surface_ || plug == local_) {
		WrapCudaDeformer *instance = (WrapCudaDeformer*) client_data;
		instance->registration_phase_ = true;
	}
}

void* WrapCudaDeformer::creator()
{
	return new WrapCudaDeformer();
}

MMatrixArray
WrapCudaDeformer::controlsMatrices(const MPointArray &vertices,
				const MIntArray &triangles_indices,
				bool inverse)
{
	unsigned int triangles_count = triangles_indices.length() / 3;
	MMatrixArray matrices(triangles_count);
	for (unsigned int i = 0; i < triangles_count; i++) {
		unsigned int index_A = triangles_indices[i * 3];
		unsigned int index_C = triangles_indices[i * 3 + 1];
		unsigned int index_B = triangles_indices[i * 3 + 2];

		MVector e1 = vertices[index_C] - vertices[index_A];
		MVector e2 = vertices[index_B] - vertices[index_A];
		MVector e1_e2_cross_product = e1 ^ e2;
		MVector e3 = e1_e2_cross_product / e1_e2_cross_product.length();

		double d_matrix[4][4] = {{e1.x, e1.y, e1.z, 0},
					{e2.x, e2.y, e2.z, 0},
					{e3.x, e3.y, e3.z, 0},
					{0,    0,    0,    1}};
		MMatrix matrix(d_matrix);
		if (inverse)
			matrices[i] = matrix.inverse();
		else
			matrices[i] = matrix;
	}
	return matrices;
};

void
WrapCudaDeformer::computeWeights(MPointArray& deformed_points,
				float local,
				double* distances,
				unsigned int deformed_points_count,
				unsigned int triangles_count,
				MPointArray& ref_vertices,
				MMatrixArray& reference_matrices) {
	MPoint point;
	MPoint triangle[3];
	double weights[deformed_points_count][triangles_count];
	double weights_sums[deformed_points_count];

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, deformed_points_count, 1),
		[&](const tbb::blocked_range<size_t> &r) { 
	for (size_t i = r.begin(); i != r.end(); ++i) {
		point = deformed_points[i];
		MPointArray contol_space_points(triangles_count);
		weights_sums[i] = 0;

		for (unsigned int j = 0; j < triangles_count; j++) {
			double distance = distances[i * triangles_count + j];

			weights[i][j] = 1 / (1 + pow(distance, local));
			weights_sums[i] += weights[i][j];

			MPoint contol_space_point = point * reference_matrices[j];
			contol_space_points[j] = contol_space_point;
		}
		points_data_[i].contol_space_points = contol_space_points;

	}
	});

	for (unsigned int i = 0; i < deformed_points_count; i++)
	{
		std::vector<double> normalised_weights;
		for (unsigned int j = 0; j < triangles_count; j++) {
			normalised_weights.push_back(weights[i][j] / weights_sums[i]);
		}
		points_data_[i].normalised_weights = normalised_weights;
	}
}

void
WrapCudaDeformer::computeWeightsCuda(MPointArray& deformed_points,
				float local,
				double* D,
				unsigned int num_points,
				unsigned int num_elems,
				MPointArray& ref_vertices) {

	double P[num_points * 3];
	MPoint iter_point;
	for (unsigned int i = 0; i < num_points; i++) {
		iter_point = deformed_points[i];
		P[i * 3] = iter_point.x;
		P[i * 3 + 1] = iter_point.y;
		P[i * 3 + 2] = iter_point.z;
	}

	unsigned int Ridx[num_elems * 3];
	for (unsigned int i = 0; i < num_elems * 3; i++) {
		Ridx[i] = triangles_vertices_[i];
	}

	unsigned int num_refs = ref_vertices.length();
	double R[num_refs * 3];
	for (unsigned int i = 0; i < num_refs; i++) {
		R[i * 3] = ref_vertices[i].x;
		R[i * 3 + 1] = ref_vertices[i].y;
		R[i * 3 + 2] = ref_vertices[i].z;
	}

	double C[num_points * num_elems * 3];

	CudaComputeControlPoints(P, Ridx, R, C, num_refs, num_points, num_elems);

	for (unsigned int i = 0; i < num_points; i++) {
		MPointArray control_points(num_elems);
		for (unsigned int j = 0; j < num_elems; j++) {
			unsigned int k = i * num_elems * 3 + j * 3;
			control_points[j].x = C[k];
			control_points[j].y = C[k + 1];
			control_points[j].z = C[k + 2];
		}
		points_data_[i].contol_space_points = control_points;
	}

	double Wnorm[num_points * num_elems];

	CudaComputeWeights(local, D, Wnorm, num_points, num_elems);

	for (unsigned int i = 0; i < num_points; i++) {
		points_data_[i].normalised_weights.resize(num_elems);
		for (unsigned int j = 0; j < num_elems; j++) {
			points_data_[i].normalised_weights[j] = (
				Wnorm[i * num_elems + j]
			);
		}
	}
}


void
WrapCudaDeformer::applyWrap(MItGeometry& iter_geo,
			MPointArray& deformed_points,
			MPointArray& driver_vertices) {

	unsigned int triangles_count = triangles_vertices_.length() / 3;
	MMatrixArray driver_matrices(triangles_count);
	driver_matrices = controlsMatrices(driver_vertices,
					triangles_vertices_,
					false);

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, deformed_points.length(), 1),
		[&](const tbb::blocked_range<size_t> &r) { 
	for (size_t i = r.begin(); i != r.end(); ++i) {
		MPoint point_deformed(0.0, 0.0, 0.0, 1.0);
		for (unsigned int j = 0; j < triangles_count; j++) {
			MPoint p = (points_data_[i].contol_space_points[j] *
						 driver_matrices[j]);
			p = p * points_data_[i].normalised_weights[j];
			point_deformed.x += p.x;
			point_deformed.y += p.y;
			point_deformed.z += p.z;
		}
		deformed_points[i] = point_deformed;
	}
	});

	iter_geo.setAllPositions(deformed_points);
}


void
WrapCudaDeformer::applyWrapCuda(MItGeometry& iter_geo,
				MPointArray& deformed_points,
				MPointArray& driver_vertices) {

	MStatus status;
	unsigned int num_points = deformed_points.length();
	CHECK_MSTATUS(status);
	unsigned int num_elems = triangles_vertices_.length() / 3;

	unsigned int DRidx[num_elems * 3];
	for (unsigned int i = 0; i < num_elems * 3; i++) {
		DRidx[i] = triangles_vertices_[i];
	}

	unsigned int num_driv = driver_vertices.length();
	double DR[num_driv * 3];
	for (unsigned int i = 0; i < num_driv; i++) {
		DR[i * 3] = driver_vertices[i].x;
		DR[i * 3 + 1] = driver_vertices[i].y;
		DR[i * 3 + 2] = driver_vertices[i].z;
	}

	double C[num_points * num_elems * 3];
	double Wnorm[num_points * num_elems];
	for (unsigned int i = 0; i < num_points; i++) {
		for (unsigned int j = 0; j < num_elems; j++) {
			unsigned int offset = i * num_elems + j;
			Wnorm[offset] = points_data_[i].normalised_weights[j];

			unsigned int k = i * num_elems * 3 + j * 3;
			C[k] = points_data_[i].contol_space_points[j].x;
			C[k + 1] = points_data_[i].contol_space_points[j].y;
			C[k + 2] = points_data_[i].contol_space_points[j].z;
		}
	}

	double P[num_points * 3];

	CudaApplyDeform(C, Wnorm, DRidx, DR, P, num_driv, num_points, num_elems);

	for (unsigned int i = 0; i < num_points; i++) {
		MPoint point_deformed(0.0, 0.0, 0.0, 1.0);
		point_deformed.x = P[i * 3];
		point_deformed.y = P[i * 3 + 1];
		point_deformed.z = P[i * 3 + 2];
		deformed_points[i] = point_deformed;
	}
	iter_geo.setAllPositions(deformed_points);
};


MStatus WrapCudaDeformer::deform(MDataBlock& block,
				MItGeometry& iter_geo,
				const MMatrix& local_to_world_matrix,
				unsigned int geometry_index)
{
	MStatus status;
	MDataHandle data_handle;

	data_handle = block.inputValue(cuda_, &status);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	bool cuda = data_handle.asBool();
	MPointArray deformed_points;
	status = iter_geo.allPositions(deformed_points);
	CHECK_MSTATUS(status);

	// Registration phase
	if (registration_phase_) {
		data_handle = block.inputValue(reference_surface_, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
	
		MObject o_reference_surface = data_handle.asMesh();
		MFnMesh reference_surface(o_reference_surface, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);

		MPointArray ref_vertices;
		status = reference_surface.getPoints(ref_vertices, MSpace::kWorld);
		CHECK_MSTATUS(status);
		MIntArray triangles_in_poly_count;
		status = reference_surface.getTriangles(triangles_in_poly_count, 
							triangles_vertices_);
		CHECK_MSTATUS(status);

		unsigned int triangles_count = triangles_vertices_.length() / 3;
		MMatrixArray reference_matrices(triangles_count);
		reference_matrices = controlsMatrices(ref_vertices,
						triangles_vertices_,
						true);

		data_handle = block.inputValue(local_, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		float local = data_handle.asFloat();

		unsigned int deformed_points_count = iter_geo.count(&status);
		CHECK_MSTATUS(status);

		//int cuda = CudaDeviceCount();
		//std::cout << "CUDA DEVS " << cuda << "\n";

		points_data_.clear();
		points_data_.resize(deformed_points_count);


		double distances[deformed_points_count * triangles_count];

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, deformed_points_count, 1),
			[&](const tbb::blocked_range<size_t> &r) { 
        	for (size_t i = r.begin(); i != r.end(); ++i) {

				MPoint point = deformed_points[i];
				MPointArray contol_space_points(triangles_count);

				for (unsigned int j = 0; j < triangles_count; j++) {
					unsigned int index_A = triangles_vertices_[j * 3];
					unsigned int index_C = triangles_vertices_[j * 3 + 1];
					unsigned int index_B = triangles_vertices_[j * 3 + 2];
					MPoint triangle[3];
					triangle[0] = ref_vertices[index_A];
					triangle[1] = ref_vertices[index_B];
					triangle[2] = ref_vertices[index_C];
					double distance = PointToTriangle(point, triangle);
					distances[i * triangles_count + j] = distance;
				}
        	}
		});
	
		if (cuda) {
			computeWeightsCuda(deformed_points, local, distances,
					deformed_points_count,
					triangles_count, ref_vertices);
		} else {
			computeWeights(deformed_points, local, distances, deformed_points_count,
				triangles_count, ref_vertices, reference_matrices);
		}

		registration_phase_ = false;
	}

	// Deformation phase
	data_handle = block.inputValue(driver_surface_, &status);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	MObject o_driver_surface = data_handle.asMesh();
	MFnMesh driver_surface(o_driver_surface, &status);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	MPointArray driver_vertices;
	status = driver_surface.getPoints(driver_vertices, MSpace::kWorld);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	
	struct timeval before, after;

	printf("Deformed vertices: %i, Control vertices %i\n",
		deformed_points.length(), triangles_vertices_.length());

	gettimeofday(&before, NULL);

	if (cuda)
		applyWrapCuda(iter_geo, deformed_points, driver_vertices);
	else
		applyWrap(iter_geo, deformed_points, driver_vertices);
 	
	gettimeofday(&after, NULL);

	printf("Runtime in microseconds for %s deform %ld\n",
		cuda ? "GPU" : "CPU",
		after.tv_usec - before.tv_usec);

	return MStatus::kSuccess;
}

MStatus WrapCudaDeformer::initialize()
{
	MStatus status;

	// Base mesh
	MFnTypedAttribute t_attr;
	reference_surface_ = t_attr.create("referenceSurface", 
					"referenceSurface", 
					MFnMeshData::kMesh);
	status = addAttribute(reference_surface_);
	CHECK_MSTATUS(status);
	status = attributeAffects(WrapCudaDeformer::reference_surface_, 
				WrapCudaDeformer::outputGeom);
	CHECK_MSTATUS(status);

	// Driver mesh
	driver_surface_ = t_attr.create("driverSurface", "driverSurface", 
					MFnMeshData::kMesh);
	status = addAttribute(driver_surface_);
	CHECK_MSTATUS(status);
	status = attributeAffects(WrapCudaDeformer::driver_surface_, 
				WrapCudaDeformer::outputGeom);
	CHECK_MSTATUS(status);
	
	// Local
	MFnNumericAttribute n_attr;
	local_ = n_attr.create("local", "local",
				MFnNumericData::kFloat, 3.5);
	status = addAttribute(local_);
	CHECK_MSTATUS(status);
	status = attributeAffects(WrapCudaDeformer::local_, 
				WrapCudaDeformer::outputGeom);

	// Enable cuda
	cuda_ = n_attr.create("cuda", "cuda",
				MFnNumericData::kBoolean, true);
	status = addAttribute(cuda_);
	CHECK_MSTATUS(status);
	status = attributeAffects(WrapCudaDeformer::cuda_,
				WrapCudaDeformer::outputGeom);

	return MStatus::kSuccess;
}
