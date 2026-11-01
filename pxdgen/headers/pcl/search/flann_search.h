// Mirror header: self-contained stand-in for <pcl/search/flann_search.h>.
//
// pcl::search::Search over a FLANN index. The index type is picked by a
// creator object nested inside the class template, which Cython cannot
// name, so construction goes through pclcompat::newFlannSearch
// (compat/search_args.h). The Search<PointT> base is flattened in.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {
namespace search {

template <typename PointT>
class FlannSearch {
public:
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setEpsilon(double eps);
    double getEpsilon();
    void setChecks(int checks);
    int getChecks();

    int nearestKSearch(const PointT& p_q, int k, std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances) const;
    int radiusSearch(const PointT& p_q, double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn) const;
};

}  // namespace search
}  // namespace pcl
