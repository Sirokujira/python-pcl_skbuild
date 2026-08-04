// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/organized_args.h.
#pragma once

#include <pcl/features/integral_image_normal.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

namespace pclcompat {

const int IINE_COVARIANCE_MATRIX = 0;
const int IINE_AVERAGE_3D_GRADIENT = 1;
const int IINE_AVERAGE_DEPTH_CHANGE = 2;
const int IINE_SIMPLE_3D_GRADIENT = 3;

void setNormalEstimationMethod(
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>& estimator,
    int method);

typedef bool (*ClusterConditionFn)(const pcl::PointXYZ& a,
                                   const pcl::PointXYZ& b,
                                   float squared_distance, void* user_data);

void setClusterCondition(
    pcl::ConditionalEuclideanClustering<pcl::PointXYZ>& clustering,
    ClusterConditionFn fn, void* user_data);

}  // namespace pclcompat
