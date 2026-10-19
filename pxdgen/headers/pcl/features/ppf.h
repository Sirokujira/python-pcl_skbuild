// Mirror header: self-contained stand-in for <pcl/features/ppf.h>.
//
// Point Pair Feature (Drost et al.) of every ordered pair of input points
// with normals: n^2 outputs, the descriptor behind PPF registration.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class PPFEstimation {
public:
    PPFEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
