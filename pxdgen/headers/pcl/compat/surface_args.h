// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/surface_args.h.
//
// What the surface wrappers cannot say in Cython: an enum nested in a
// class template, an Eigen camera matrix, and two algorithms that take
// and return a pcl::PolygonMesh (whose vertex cloud is a PCLPointCloud2
// blob). Meshes cross as a cloud plus the flat (indices, counts) pair
// pcl/compat/mesh_args.h already uses for mesh I/O.
#pragma once

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/bilateral_upsampling.h>
#include <pcl/surface/organized_fast_mesh.h>

namespace pclcompat {

const int ORGANIZED_MESH_TRIANGLE_RIGHT_CUT = 0;
const int ORGANIZED_MESH_TRIANGLE_LEFT_CUT = 1;
const int ORGANIZED_MESH_TRIANGLE_ADAPTIVE_CUT = 2;
const int ORGANIZED_MESH_QUAD_MESH = 3;

void setOrganizedMeshTriangulation(
    pcl::OrganizedFastMesh<pcl::PointXYZ>& mesh, int type);

// Row-major 3x3 camera matrix.
void setBilateralProjection(
    pcl::BilateralUpsampling<pcl::PointXYZRGBA, pcl::PointXYZRGBA>& filter,
    const float* matrix9);

// Triangulates polygon by polygon and checks each result; returns -1
// when all are right, else the index of the first polygon PCL got wrong.
int earClipping(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                const std::vector<int>& indices,
                const std::vector<int>& counts,
                std::vector<int>& out_indices,
                std::vector<int>& out_counts);

void removeUnusedVertices(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                          const std::vector<int>& indices,
                          const std::vector<int>& counts,
                          pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                          std::vector<int>& out_indices,
                          std::vector<int>& out_counts,
                          std::vector<int>& out_kept);

}  // namespace pclcompat
