// Mirror header: self-contained stand-in for
// <pcl/features/linear_least_squares_normal.h>.
//
// Normals for an organized cloud from a least-squares plane over each
// pixel's depth neighbourhood -- an alternative to the integral-image
// estimator.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class LinearLeastSquaresNormalEstimation {
public:
    LinearLeastSquaresNormalEstimation();

    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setNormalSmoothingSize(float normal_smoothing_size);
    void setDepthDependentSmoothing(bool use_depth_dependent_smoothing);
    void setMaxDepthChangeFactor(float max_depth_change_factor);
};

}  // namespace pcl
