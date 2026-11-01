// Mirror header: self-contained stand-in for <pcl/features/shot_lrf.h>.
//
// The SHOT local reference frame on its own: three repeatable axes per
// point from the weighted covariance of its neighbourhood.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class SHOTLocalReferenceFrameEstimation {
public:
    SHOTLocalReferenceFrameEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
