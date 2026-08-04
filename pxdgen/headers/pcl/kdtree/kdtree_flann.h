// Mirror header: self-contained stand-in for <pcl/kdtree/kdtree_flann.h>.
//
// The real k_indices parameter is pcl::Indices; since PCL 1.11 that is
// std::vector<pcl::index_t> with index_t = int, so std::vector<int> is
// the same type. See filters/voxel_grid.h for the flattening rationale.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class KdTreeFLANN {
public:
    KdTreeFLANN();
    KdTreeFLANN(bool sorted);

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setEpsilon(float eps);
    float getEpsilon() const;
    void setSortedResults(bool sorted);
    void setMinPts(int min_pts);
    int getMinPts() const;

    int nearestKSearch(const PointT& point,
                       unsigned int k,
                       std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances) const;

    int radiusSearch(const PointT& point,
                     double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn = 0) const;
};

}  // namespace pcl
