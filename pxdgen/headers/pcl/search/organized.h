// Mirror header: self-contained stand-in for <pcl/search/organized.h>.
//
// Neighbour search that uses the pixel grid of an organized cloud from a
// projective camera instead of building a tree: setInputCloud estimates
// the camera matrix and returns false when the cloud is not organized or
// not projective. The Search<PointT> base is flattened in; the indices
// argument of setInputCloud is left to its default.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {
namespace search {

template <typename PointT>
class OrganizedNeighbor {
public:
    OrganizedNeighbor(bool sorted_results, float eps, unsigned int pyramid_level);

    bool setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    bool isValid() const;

    int nearestKSearch(const PointT& p_q, int k, std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances) const;
    int radiusSearch(const PointT& p_q, double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn) const;
};

}  // namespace search
}  // namespace pcl
