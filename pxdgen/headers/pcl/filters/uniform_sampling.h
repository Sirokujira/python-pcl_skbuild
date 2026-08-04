// Mirror header: self-contained stand-in for
// <pcl/filters/uniform_sampling.h>.
//
// python-pcl reaches this through pcl/pxi/KeyPoint/UniformSampling.pxi
// because PCL used to ship it under keypoints/; since 1.9 it is a filter,
// which is why it lives here.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class UniformSampling {
public:
    UniformSampling();
    UniformSampling(bool extract_removed_indices);

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setRadiusSearch(double radius);
};

}  // namespace pcl
