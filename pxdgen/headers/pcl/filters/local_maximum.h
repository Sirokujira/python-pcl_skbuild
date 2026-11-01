// Mirror header: self-contained stand-in for <pcl/filters/local_maximum.h>.
//
// Drops every point that has a higher (z) neighbour within the radius,
// keeping the local maxima.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class LocalMaximum {
public:
    LocalMaximum();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setRadius(float radius);
    float getRadius() const;
};

}  // namespace pcl
