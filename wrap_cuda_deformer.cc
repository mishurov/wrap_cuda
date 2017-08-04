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
WrapCudaDeformer::computeWeights(MItGeometry& iter_geo,
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

	iter_geo.reset();
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		point = iter_geo.position();
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
		i++;
	}

	iter_geo.reset();
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next())
	{
		std::vector<double> normalised_weights;
		for (unsigned int j = 0; j < triangles_count; j++) {
			normalised_weights.push_back(weights[i][j] / weights_sums[i]);
		}
		points_data_[i].normalised_weights = normalised_weights;
		i++;
	}
}

void
WrapCudaDeformer::computeWeightsCuda(MItGeometry& iter_geo,
								 	float local,
								 	double* distances,
								 	unsigned int deformed_points_count,
								 	unsigned int triangles_count,
								 	MPointArray& ref_vertices,
								 	MMatrixArray& reference_matrices) {

	// convert to the primitive types
	double deformed_points[deformed_points_count * 3];
	iter_geo.reset();
	MPoint iter_point;
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		iter_point = iter_geo.position();
		deformed_points[i * 3] = iter_point.x;
		deformed_points[i * 3 + 1] = iter_point.y;
		deformed_points[i * 3 + 2] = iter_point.z;
		i++;
	}

	unsigned int ref_vertices_count = ref_vertices.length();
	double ref_vertices_cu[ref_vertices_count * 3];
	for (unsigned int i = 0; i < ref_vertices_count; i++) {
		ref_vertices_cu[i * 3] = ref_vertices[i].x;
		ref_vertices_cu[i * 3 + 1] = ref_vertices[i].y;
		ref_vertices_cu[i * 3 + 2] = ref_vertices[i].z;
	}

	unsigned int triangles_vertices[triangles_count * 3];
	double reference_matrices_cu[triangles_count * 3 * 3];
	for (unsigned int i = 0; i < triangles_count; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			triangles_vertices[i] = triangles_vertices_[i * 3 + j];
			for (unsigned int k = 0; k < 3; k++) {
				reference_matrices_cu[(i * 3 + j) * 3 + k] = reference_matrices[i](j, k);
			}
		}
	}

	double normalised_weights[deformed_points_count * triangles_count];
	double contol_space_points[deformed_points_count * triangles_count * 3];

	CudaComputeWeights(
		normalised_weights,
		contol_space_points,
		local,
		distances,
		deformed_points,
		deformed_points_count,
		triangles_count,
		triangles_vertices,
		ref_vertices_cu,
		ref_vertices_count,
		reference_matrices_cu
	);

	for (unsigned int i = 0; i < deformed_points_count; i++) {
		points_data_[i].normalised_weights.resize(triangles_count);
		for (unsigned int j = 0; j < triangles_count; j++) {
			points_data_[i].normalised_weights[j] = (
				normalised_weights[i * triangles_count + j]
			);
			/*
			points_data_[i].contol_space_points[j].x = contol_space_points[i * j];
			points_data_[i].contol_space_points[j].y = contol_space_points[i * j + 1];
			points_data_[i].contol_space_points[j].z = contol_space_points[i * j + 2];
			*/
		}
	}

	iter_geo.reset();
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		MPoint point = iter_geo.position();
		MPointArray contol_space_points(triangles_count);
		for (unsigned int j = 0; j < triangles_count; j++) {
			MPoint contol_space_point = point * reference_matrices[j];
			contol_space_points[j] = contol_space_point;
		}
		points_data_[i].contol_space_points = contol_space_points;
		i++;
	}
}


void
WrapCudaDeformer::applyWrap(MItGeometry& iter_geo,
							MPointArray& driver_vertices) {

	unsigned int triangles_count = triangles_vertices_.length() / 3;
	MMatrixArray driver_matrices(triangles_count);
	driver_matrices = controlsMatrices(driver_vertices,
									   triangles_vertices_,
									   false);
	iter_geo.reset();
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		MPoint point_deformed(0.0, 0.0, 0.0, 1.0);
		for (unsigned int j = 0; j < triangles_count; j++) {
			MPoint cp = (points_data_[i].contol_space_points[j] *
						 driver_matrices[j]);
			cp = cp * points_data_[i].normalised_weights[j];
			point_deformed.x += cp.x;
			point_deformed.y += cp.y;
			point_deformed.z += cp.z;
		}
		iter_geo.setPosition(point_deformed);
		i++;
	}
}


void
WrapCudaDeformer::applyWrapCuda(MItGeometry& iter_geo,
								MPointArray& driver_vertices) {

	MStatus status;
	unsigned int deformed_points_count = iter_geo.count(&status);
	CHECK_MSTATUS(status);
	unsigned int triangles_count = triangles_vertices_.length() / 3;

	double points[deformed_points_count * 3];
	iter_geo.reset();
	MPoint iter_point;
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		iter_point = iter_geo.position();
		points[i * 3] = iter_point.x;
		points[i * 3 + 1] = iter_point.y;
		points[i * 3 + 2] = iter_point.z;
		i++;
	}

	unsigned int triangles_indices[triangles_count * 3];
	for (unsigned int i = 0; i < triangles_count * 3; i++) {
		triangles_indices[i] = triangles_vertices_[i];
	}

	unsigned int driver_vertices_count = driver_vertices.length();
	double driver_vertices_cu[driver_vertices_count * 3];
	for (unsigned int i = 0; i < driver_vertices_count; i++) {
		driver_vertices_cu[i * 3] = driver_vertices[i].x;
		driver_vertices_cu[i * 3 + 1] = driver_vertices[i].y;
		driver_vertices_cu[i * 3 + 2] = driver_vertices[i].z;
	}

	double cs_points[deformed_points_count * triangles_count * 3];
	double normalised_weights[deformed_points_count * triangles_count];
	for (unsigned int i = 0; i < deformed_points_count; i++) {
		for (unsigned int j = 0; j < triangles_count; j++) {
			unsigned int offset = i * triangles_count + j;
			normalised_weights[offset] = points_data_[i].normalised_weights[j];
			cs_points[offset] = points_data_[i].contol_space_points[j].x;
			cs_points[offset + 1] = points_data_[i].contol_space_points[j].y;
			cs_points[offset + 2] = points_data_[i].contol_space_points[j].z;
		}
	}

	MMatrixArray driver_matrices(triangles_count);
	driver_matrices = controlsMatrices(driver_vertices,
									   triangles_vertices_,
									   false);

	double mats[triangles_count*9];
	/*
	for (unsigned int i = 0; i < triangles_count*9; i++) {
		unsigned int idx = i * 9;
		mats[idx] = driver_matrices[i](1, 1);
		mats[idx + 1] = driver_matrices[i](1, 2);
		mats[idx + 2] = driver_matrices[i](1, 3);
		mats[idx + 3] = driver_matrices[i](2, 1);
		mats[idx + 4] = driver_matrices[i](2, 2);
		mats[idx + 5] = driver_matrices[i](2, 3);
		mats[idx + 6] = driver_matrices[i](3, 1);
		mats[idx + 7] = driver_matrices[i](3, 2);
		mats[idx + 8] = driver_matrices[i](3, 3);
	}
	*/

	double out_points[deformed_points_count * 3];
	CudaApplyDeform(
		out_points,
		cs_points,
		points,
		deformed_points_count,
		triangles_indices,
		triangles_count,
		driver_vertices_cu,
		driver_vertices_count,
		normalised_weights,
		mats
	);

	iter_geo.reset();
	for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
		MPoint point_deformed(0.0, 0.0, 0.0, 1.0);
		point_deformed.x = out_points[i * 3];
		point_deformed.y = out_points[i * 3 + 1];
		point_deformed.z = out_points[i * 3 + 2];
		/*
		for (unsigned int j = 0; j < triangles_count; j++) {
			unsigned int idx = j * 9;
			double d_matrix[4][4] = {{mats[idx], mats[idx+1], mats[idx+2], 0}, 
									 {mats[idx+3], mats[idx+4], mats[idx+5], 0}, 
									 {mats[idx+6], mats[idx+7], mats[idx+8], 0}, 
									 {0,    0,    0,    1}};
			MMatrix matrix(d_matrix);

			MPoint cp = (points_data_[i].contol_space_points[j] *
						 matrix);

			cp = cp * points_data_[i].normalised_weights[j];
			point_deformed.x += cp.x;
			point_deformed.y += cp.y;
			point_deformed.z += cp.z;
		}
		*/
		iter_geo.setPosition(point_deformed);
		i++;
	}
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


		iter_geo.reset();
		double distances[deformed_points_count * triangles_count];
		for (unsigned int i = 0; !iter_geo.isDone(); iter_geo.next()) {
			MPoint point = iter_geo.position();
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
			i++;
		}

		if (cuda) {
			computeWeightsCuda(iter_geo, local, distances,
							   deformed_points_count,
							   triangles_count, ref_vertices,
							   reference_matrices);
		} else {
			computeWeights(iter_geo, local, distances, deformed_points_count,
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

	if (cuda)
		applyWrapCuda(iter_geo, driver_vertices);
	else
		applyWrap(iter_geo, driver_vertices);

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
	driver_surface_ = t_attr.create("driverSurface", 
									"driverSurface", 
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
