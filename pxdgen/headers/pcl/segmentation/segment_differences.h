// Mirror header: self-contained stand-in for <pcl/segmentation/segment_differences.h>.
//
// The points of the input that have no target point within a distance;
// the threshold is the SQUARED distance in PCL.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class SegmentDifferences {
public:
    SegmentDifferences();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setTargetCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setDistanceThreshold(double sqr_threshold);
    double getDistanceThreshold();
    void segment(PointCloud<PointT>& output);
};

}  // namespace pcl
