// Mirror header: self-contained stand-in for <pcl/filters/grid_minimum.h>.
//
// Keeps the lowest point of every cell of a 2-D grid over x/y: a ground
// estimate for aerial or terrestrial scans. The resolution is a
// constructor argument in PCL (no default constructor).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class GridMinimum {
public:
    GridMinimum(float resolution);

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setResolution(float resolution);
    float getResolution();
};

}  // namespace pcl
