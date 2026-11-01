// Mirror header: self-contained stand-in for <pcl/filters/bilateral.h>.
//
// Edge-preserving smoothing of the INTENSITY channel (PointXYZI): each
// value is replaced by a weighted mean of its neighbours' within
// half-size, weights falling off with spatial and intensity distance.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class BilateralFilter {
public:
    BilateralFilter();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setHalfSize(double sigma_s);
    double getHalfSize() const;
    void setStdDev(double sigma_r);
    double getStdDev() const;
};

}  // namespace pcl
