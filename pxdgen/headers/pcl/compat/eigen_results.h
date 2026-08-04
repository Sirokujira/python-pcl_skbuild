// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/eigen_results.h (reached as
// "pcl/compat/eigen_results.h" because CMake adds src/ to the includes).
//
// The real header includes Eigen and PCL; this one names only what
// Cython has to see, so pxd generation still runs on a machine with
// neither installed.
#pragma once

#include <vector>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ndt.h>

namespace pclcompat {

void finalTransformation(
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>& reg,
    float* out16);
void finalTransformation(
    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ>& reg,
    float* out16);
void finalTransformation(
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>& reg,
    float* out16);
void finalTransformation(
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>& reg,
    float* out16);

void massCenter(pcl::MomentOfInertiaEstimation<pcl::PointXYZ>& est,
                float* out3);
void axisAlignedBoundingBox(
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ>& est, float* out6);
void orientedBoundingBox(pcl::MomentOfInertiaEstimation<pcl::PointXYZ>& est,
                         float* out18);
void eigenValues(pcl::MomentOfInertiaEstimation<pcl::PointXYZ>& est,
                 float* out3);
void eigenVectors(pcl::MomentOfInertiaEstimation<pcl::PointXYZ>& est,
                  float* out9);

int occupiedVoxelCenters(
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>& octree,
    std::vector<pcl::PointXYZ>& out);

}  // namespace pclcompat
