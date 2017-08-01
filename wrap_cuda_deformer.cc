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

MTypeId WrapCudaDeformer::id(kWrapCudaDeformerID);
MObject WrapCudaDeformer::reference_surface_;
MObject WrapCudaDeformer::driver_surface_;

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
  if (plug == reference_surface_) {
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
                                   const MIntArray &triangles_indices)
{
  unsigned int triangles_count = triangles_indices.length() / 3;
  MMatrixArray matrices(triangles_count);
  for (unsigned int i = 0; i < triangles_count; i++) {
    unsigned int index_A = triangles_indices[i*3];
    unsigned int index_C = triangles_indices[i*3+1];
    unsigned int index_B = triangles_indices[i*3+2];

    MVector e1 = vertices[index_C] - vertices[index_A];
    MVector e2 = vertices[index_B] - vertices[index_A];
    MVector e1_e2_cross_product = e1 ^ e2;
    MVector e3 = e1_e2_cross_product / e1_e2_cross_product.length();

    double d_matrix[4][4] = {{e1.x, e1.y, e1.z, 0}, 
                             {e2.x, e2.y, e2.z, 0}, 
                             {e3.x, e3.y, e3.z, 0}, 
                             {0,    0,    0,    1}};
    MMatrix matrix(d_matrix);
    matrices[i] = matrix;
  }
  return matrices;
};

MStatus WrapCudaDeformer::deform(MDataBlock& block,
                                 MItGeometry& iter_geo,
                                 const MMatrix& local_to_world_matrix,
                                 unsigned int geometry_index)
{
  
  MStatus status;
  MDataHandle data_handle;
  unsigned int i;

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

    // Calculate control elements' local space inverse matrices
    unsigned int triangles_count = triangles_vertices_.length() / 3;
    MMatrixArray reference_matrices(triangles_count);
    reference_matrices = controlsMatrices(ref_vertices, triangles_vertices_);
    
    // Calculate for every point of deformed object
    // distance weights and control elements' space coordiantes
    // TODO: "local" as node attribute;
    double local = 3.5;
    
    MPoint point;
	MPoint triangle[3];

    unsigned int deformed_points_count = iter_geo.count(&status);
    CHECK_MSTATUS(status);
    double weights[deformed_points_count][triangles_count];
    double weights_sums[deformed_points_count];
    points_data_.clear();
    points_data_.resize(deformed_points_count);
    
    iter_geo.reset();
    for (i = 0; !iter_geo.isDone(); iter_geo.next())
    {
      point = iter_geo.position();
      MPointArray contol_space_points(triangles_count);
      weights_sums[i] = 0;

      for (unsigned int e = 0; e < triangles_count; e++) {
        unsigned int index_A = triangles_vertices_[e*3];
        unsigned int index_C = triangles_vertices_[e*3+1];
        unsigned int index_B = triangles_vertices_[e*3+2];

		triangle[0] = ref_vertices[index_A];
		triangle[1] = ref_vertices[index_B];
		triangle[2] = ref_vertices[index_C];
        double distance = PointToTriangle(point, triangle);

        weights[i][e] = 1 / (1 + pow(distance, local));
        weights_sums[i] += weights[i][e];

        MPoint 
        contol_space_point = point * reference_matrices[e].inverse();
        contol_space_points[e] = contol_space_point;
      }
      points_data_[i].contol_space_points = contol_space_points;
      i++;
    }

    iter_geo.reset();
    for (i = 0; !iter_geo.isDone(); iter_geo.next())
    {
      std::vector<double> normalized_weights;
      normalized_weights.clear();
      for (unsigned int e = 0; e < triangles_count; e++) {
        normalized_weights.push_back(weights[i][e] / weights_sums[i]);
      }
      points_data_[i].normalized_weights = normalized_weights;
      i++;
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

  unsigned int triangles_count = triangles_vertices_.length() / 3;
  MMatrixArray driver_matrices(triangles_count);
  driver_matrices = controlsMatrices(driver_vertices, triangles_vertices_);

  iter_geo.reset();
  for (i = 0; !iter_geo.isDone(); iter_geo.next())
  {
    MPoint point_deformed(0.0, 0.0, 0.0, 1.0);
    for (unsigned int e = 0; e < triangles_count; e++) {
        MPoint cp = points_data_[i].contol_space_points[e] * driver_matrices[e];
        cp = cp * points_data_[i].normalized_weights[e];
        point_deformed.x += cp.x;
        point_deformed.y += cp.y;
        point_deformed.z += cp.z;
    }
    iter_geo.setPosition(point_deformed);
    i++;
  }
 
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

  return MStatus::kSuccess;
}
