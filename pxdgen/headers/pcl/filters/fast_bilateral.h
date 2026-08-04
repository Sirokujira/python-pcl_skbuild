// Mirror header: self-contained stand-in for
// <pcl/filters/fast_bilateral.h>.
//
// Edge-preserving smoothing for ORGANIZED clouds — it works on the depth
// image, so an unorganized cloud is a no-op.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class FastBilateralFilter {
public:
    FastBilateralFilter();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setSigmaS(float sigma_s);
    float getSigmaS() const;
    void setSigmaR(float sigma_r);
    float getSigmaR() const;
};

}  // namespace pcl
