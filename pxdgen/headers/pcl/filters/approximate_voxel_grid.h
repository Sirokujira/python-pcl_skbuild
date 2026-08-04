// Mirror header: self-contained stand-in for
// <pcl/filters/approximate_voxel_grid.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ApproximateVoxelGrid {
public:
    ApproximateVoxelGrid();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setLeafSize(float lx, float ly, float lz);
    void setDownsampleAllData(bool downsample);
    bool getDownsampleAllData() const;
};

}  // namespace pcl
