// Mirror header: self-contained stand-in for the free functions in
// <pcl/filters/filter.h>. `Indices` is std::vector<index_t>, and index_t
// is int on every supported PCL.
#pragma once

#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
void removeNaNFromPointCloud(const PointCloud<PointT>& cloud_in,
                             PointCloud<PointT>& cloud_out,
                             std::vector<int>& index);

}  // namespace pcl
