// Mirror header: self-contained stand-in for
// <pcl/segmentation/extract_clusters.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class EuclideanClusterExtraction {
public:
    EuclideanClusterExtraction();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setClusterTolerance(double tolerance);
    double getClusterTolerance() const;
    void setMinClusterSize(int min_cluster_size);
    int getMinClusterSize() const;
    void setMaxClusterSize(int max_cluster_size);
    int getMaxClusterSize() const;

    void extract(std::vector<PointIndices>& clusters);
};

}  // namespace pcl
