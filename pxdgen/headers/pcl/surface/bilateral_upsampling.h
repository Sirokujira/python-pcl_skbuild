// Mirror header: self-contained stand-in for
// <pcl/surface/bilateral_upsampling.h>.
//
// Fills depth holes in an organized RGB-D cloud with a bilateral filter
// steered by colour, then re-projects every pixel through the camera
// matrix. The matrix is an Eigen::Matrix3f and goes through the
// pcl/compat/surface_args.h shim; the CloudSurfaceProcessing base is
// flattened in.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class BilateralUpsampling {
public:
    BilateralUpsampling();

    // Inherited entry points (PCLBase / CloudSurfaceProcessing).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void process(PointCloud<PointOutT>& output);

    void setWindowSize(int window_size);
    int getWindowSize() const;
    void setSigmaColor(const float& sigma_color);
    float getSigmaColor() const;
    void setSigmaDepth(const float& sigma_depth);
    float getSigmaDepth() const;
};

}  // namespace pcl
