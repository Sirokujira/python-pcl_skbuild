// Mirror header: self-contained stand-in for <pcl/filters/median_filter.h>.
//
// Median of each point's depth over a window of its ORGANIZED neighbours
// (the cloud must keep its sensor's width x height layout).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class MedianFilter {
public:
    MedianFilter();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setWindowSize(int window_size);
    int getWindowSize() const;
    void setMaxAllowedMovement(float max_allowed_movement);
    float getMaxAllowedMovement() const;
};

}  // namespace pcl
