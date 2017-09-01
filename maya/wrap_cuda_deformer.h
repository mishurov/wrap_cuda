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

#ifndef MAYAPLUGIN_WRAPCUDADEFORMER_H_
#define MAYAPLUGIN_WRAPCUDADEFORMER_H_

#include <math.h>
#include <vector>

#include <maya/MPxDeformerNode.h> 
#include <maya/MItGeometry.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMeshData.h>
#include <maya/MTypeId.h>
#include <maya/MDataBlock.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MCallbackIdArray.h>
#include <maya/MNodeMessage.h>

#include <maya/MThreadUtils.h>
#include <tbb/tbb.h>


#include "../utils/point_to_triangle.h"
#include "../cuda/cuda_ops.h"

#define kWrapCudaDeformerID 0x8000C

struct PointData {
	std::vector<double> normalised_weights;
	MPointArray contol_space_points;
};

class WrapCudaDeformer : public MPxDeformerNode
{
public:
	static MTypeId id;
	static MObject reference_surface_;
	static MObject driver_surface_;
	static MObject local_;
	static MObject cuda_;

	WrapCudaDeformer();
	virtual ~WrapCudaDeformer();
	virtual void postConstructor();
	static void* creator();
	static MStatus initialize();

	virtual MStatus deform(MDataBlock&,
				MItGeometry&,
				const MMatrix&,
				unsigned int);

private:
	MCallbackIdArray callback_ids_;
	bool registration_phase_;
	MIntArray triangles_vertices_;
	std::vector<PointData> points_data_;

	static void registrationCallback(MObject&, MPlug&, void*);
	static MMatrixArray controlsMatrices(const MPointArray &, 
					const MIntArray &,
					bool inverse);
	void computeWeights(MPointArray& deformed_points,
			float local,
			double* distances,
			unsigned int deformed_points_count,
			unsigned int triangles_count,
			MPointArray& ref_vertices,
			MMatrixArray& reference_matrices);
	void computeWeightsCuda(MPointArray& deformed_points,
				float local,
				double* distances,
				unsigned int deformed_points_count,
				unsigned int triangles_count,
				MPointArray& ref_vertices);
	void applyWrap(MItGeometry& iter_geo,
			MPointArray& deformed_points,
			MPointArray& driver_vertices);
	void applyWrapCuda(MItGeometry& iter_geo,
			MPointArray& deformed_points,
			MPointArray& driver_vertices);
};

#endif  // MAYAPLUGIN_WRAPCUDADEFORMER_H_
