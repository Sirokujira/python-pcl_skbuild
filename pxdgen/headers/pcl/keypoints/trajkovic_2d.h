// Mirror header: self-contained stand-in for <pcl/keypoints/trajkovic_2d.h>.
//
// Trajkovic and Hedley's fast corner detector on the intensity image of
// an organized cloud. The ComputationMethod enum (four or eight corners)
// goes through pcl/compat/keypoint_args.h.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class TrajkovicKeypoint2D {
public:
    TrajkovicKeypoint2D();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setWindowSize(int window_size);
    int getWindowSize() const;
    void setFirstThreshold(float threshold);
    float getFirstThreshold() const;
    void setSecondThreshold(float threshold);
    float getSecondThreshold() const;
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
