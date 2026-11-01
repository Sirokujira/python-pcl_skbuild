// Mirror header: self-contained stand-in for <pcl/keypoints/agast_2d.h>.
//
// AGAST corners (Mair et al.) on the intensity image of an organized
// cloud; keypoints come back as pixel coordinates (PointUV).
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class AgastKeypoint2D {
public:
    AgastKeypoint2D();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setThreshold(const double threshold);
    double getThreshold();
    void setMaxKeypoints(const unsigned int nr_max_keypoints);
    unsigned int getMaxKeypoints();
    void setMaxDataValue(const double bmax);
    double getMaxDataValue();
    void setNonMaxSuppression(const bool enabled);
    bool getNonMaxSuppression();
};

}  // namespace pcl
