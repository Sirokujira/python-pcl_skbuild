// C++ shim: the plain names behind pcl::Histogram<N> (an integer
// template argument Cython cannot pass), SUSANKeypoint's intensity
// accessor template argument, and CRHEstimation::setCentroid (an Eigen
// vector). See pxdgen/headers/pcl/compat/feature_types.h for the mirror.
#pragma once

#include <pcl/features/crh.h>
#include <pcl/keypoints/susan.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Histogram153 = pcl::Histogram<153>;
using Histogram90 = pcl::Histogram<90>;
using SUSANKeypointRGB = pcl::SUSANKeypoint<pcl::PointXYZRGB, pcl::PointXYZ>;

inline void setCRHCentroid(
    pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, Histogram90>& estimator,
    float x, float y, float z) {
    Eigen::Vector4f centroid(x, y, z, 0.0f);
    estimator.setCentroid(centroid);
}

}  // namespace pclcompat
