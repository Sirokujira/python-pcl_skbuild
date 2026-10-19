// Mirror header: self-contained stand-in for <pcl/features/brisk_2d.h>.
//
// BRISK binary descriptors (512 bits) at BriskKeypoint2D keypoints.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT, typename KeypointT>
class BRISK2DEstimation {
public:
    BRISK2DEstimation();

    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setKeypoints(const std::shared_ptr<PointCloud<KeypointT>>& keypoints);
    void compute(PointCloud<PointOutT>& output);

    void setRotationInvariance(const bool enable);
    void setScaleInvariance(const bool enable);
};

}  // namespace pcl
