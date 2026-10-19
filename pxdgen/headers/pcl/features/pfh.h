// Mirror header: self-contained stand-in for <pcl/features/pfh.h>.
//
// Point Feature Histogram: the 125-bin ancestor of FPFH, exact but
// O(k^2) per point. Needs normals and a neighbourhood size.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class PFHEstimation {
public:
    PFHEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setMaximumCacheSize(unsigned int cache_size);
    unsigned int getMaximumCacheSize();
    void setUseInternalCache(bool use_cache);
    bool getUseInternalCache();
};

}  // namespace pcl
