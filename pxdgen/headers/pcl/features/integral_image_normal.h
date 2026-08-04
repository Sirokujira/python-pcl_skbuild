// Mirror header: self-contained stand-in for
// <pcl/features/integral_image_normal.h>.
//
// Normals straight from a depth image: much faster than
// NormalEstimation, but it needs an ORGANIZED cloud (build one with
// PointCloud.from_organized_array).
//
// setNormalEstimationMethod takes an enum nested in this class template,
// which Cython cannot name — it goes through pcl/compat/organized_args.h.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class IntegralImageNormalEstimation {
public:
    IntegralImageNormalEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setMaxDepthChangeFactor(float max_depth_change_factor);
    void setNormalSmoothingSize(float normal_smoothing_size);
    void setDepthDependentSmoothing(bool use_depth_dependent_smoothing);
    void setRectSize(int width, int height);
    void setViewPoint(float vpx, float vpy, float vpz);
};

}  // namespace pcl
