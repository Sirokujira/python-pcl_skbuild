// Mirror header: self-contained stand-in for
// <pcl/segmentation/conditional_euclidean_clustering.h>.
//
// Euclidean clustering where the caller decides whether two neighbours
// belong together — the condition is a std::function, so it is set
// through pcl/compat/organized_args.h.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ConditionalEuclideanClustering {
public:
    ConditionalEuclideanClustering();
    ConditionalEuclideanClustering(bool extract_removed_clusters);

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setClusterTolerance(float cluster_tolerance);
    float getClusterTolerance();
    void setMinClusterSize(int min_cluster_size);
    int getMinClusterSize();
    void setMaxClusterSize(int max_cluster_size);
    int getMaxClusterSize();

    void segment(std::vector<PointIndices>& clusters);
};

}  // namespace pcl
