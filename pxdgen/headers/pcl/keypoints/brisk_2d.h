// Mirror header: self-contained stand-in for <pcl/keypoints/brisk_2d.h>.
//
// BRISK keypoints (Leutenegger et al.): AGAST corners over a scale-space
// pyramid of the intensity image, with sub-pixel and sub-scale
// refinement. PointWithScale output; the intensity-selector parameter is
// left to its default.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class BriskKeypoint2D {
public:
    BriskKeypoint2D(int octaves, int threshold);

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setThreshold(const int threshold);
    int getThreshold();
    void setOctaves(const int octaves);
    int getOctaves();
    void setRemoveInvalid3DKeypoints(bool remove);
    bool getRemoveInvalid3DKeypoints();
};

}  // namespace pcl
