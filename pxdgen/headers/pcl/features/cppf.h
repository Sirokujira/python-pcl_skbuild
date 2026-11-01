// Mirror header: self-contained stand-in for <pcl/features/cppf.h>.
//
// Colour Point Pair Feature: ten geometric and colour features per
// ordered pair.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class CPPFEstimation {
public:
    CPPFEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
