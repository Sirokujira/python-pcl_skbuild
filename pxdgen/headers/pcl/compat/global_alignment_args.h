// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/global_alignment_args.h (reached as
// "pcl/compat/global_alignment_args.h" because CMake adds src/ to the
// includes).
//
// The real header names Eigen, FPFH feature clouds and PCL's
// SampleConsensusPrerejective; this one names only the single entry
// point Cython calls, so pxd generation still runs without PCL or Eigen
// installed.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

bool globalAlignment(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
                     const std::vector<float>& source_features,
                     const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
                     const std::vector<float>& target_features,
                     int max_iterations, int number_of_samples,
                     int correspondence_randomness,
                     float similarity_threshold,
                     float max_correspondence_distance,
                     float inlier_fraction,
                     pcl::PointCloud<pcl::PointXYZ>& out_aligned,
                     float* out_matrix16,
                     std::vector<int>& out_inliers,
                     double* out_fitness);

}  // namespace pclcompat
