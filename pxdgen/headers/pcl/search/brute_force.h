// Mirror header: self-contained stand-in for <pcl/search/brute_force.h>.
//
// Exhaustive neighbour search: no index at all, every query scans the
// cloud. Exact, and faster than a tree for small clouds or few queries.
// The Search<PointT> base is flattened in.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {
namespace search {

template <typename PointT>
class BruteForce {
public:
    BruteForce(bool sorted_results);

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    int nearestKSearch(const PointT& p_q, int k, std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances) const;
    int radiusSearch(const PointT& p_q, double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn) const;
};

}  // namespace search
}  // namespace pcl
