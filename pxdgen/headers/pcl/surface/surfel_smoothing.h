// Mirror header: self-contained stand-in for
// <pcl/surface/surfel_smoothing.h>.
//
// Smooths points AND normals together, each surfel pulled towards the
// tangent planes of its neighbours. The outputs are shared_ptr
// out-parameters, exactly as PCL takes them.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointNT>
class SurfelSmoothing {
public:
    explicit SurfelSmoothing(float a_scale);

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setInputNormals(std::shared_ptr<PointCloud<PointNT>>& a_normals);
    void computeSmoothedCloud(
        std::shared_ptr<PointCloud<PointT>>& output_positions,
        std::shared_ptr<PointCloud<PointNT>>& output_normals);
};

}  // namespace pcl
