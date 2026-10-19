// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/hypotheses_args.h.
//
// Scene and models are plain shared_ptr clouds; the models are already
// placed in the scene (a recognizer's transforms applied). Each function
// fills one int per model: 1 when the hypothesis was kept.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

void greedyVerification(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& scene,
    const std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& models,
    float resolution, float inlier_threshold, float occlusion_threshold,
    float lambda, std::vector<int>& out_mask);

void papazovVerification(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& scene,
    const std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& models,
    float resolution, float inlier_threshold, float occlusion_threshold,
    float conflict_threshold, float support_threshold,
    float penalty_threshold, std::vector<int>& out_mask);

}  // namespace pclcompat
