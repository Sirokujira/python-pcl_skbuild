// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/recognition_args.h (reached as
// "pcl/compat/recognition_args.h" because CMake adds src/ to the
// includes).
//
// The real header includes Eigen, pcl::Correspondence and both grouping
// algorithms; this one names only what Cython has to see, so pxd
// generation still runs on a machine with none of them installed.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

int geometricConsistencyGrouping(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& model,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& scene,
    const std::vector<int>& model_indices,
    const std::vector<int>& scene_indices,
    const std::vector<float>& distances,
    double gc_size, int gc_threshold,
    std::vector<float>& out_transforms,
    std::vector<int>& out_corr_counts,
    std::vector<int>& out_corr_pairs);

int hough3DGrouping(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& model,
                    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& scene,
                    const std::vector<int>& model_indices,
                    const std::vector<int>& scene_indices,
                    const std::vector<float>& distances,
                    float rf_radius, double bin_size, double threshold,
                    std::vector<float>& out_transforms,
                    std::vector<int>& out_corr_counts,
                    std::vector<int>& out_corr_pairs);

}  // namespace pclcompat
