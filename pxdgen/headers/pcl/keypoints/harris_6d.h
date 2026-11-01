// Mirror header: self-contained stand-in for <pcl/keypoints/harris_6d.h>.
//
// Harris corners in the joint space of geometry and intensity: the
// covariance of normals AND intensity gradients around each point. The
// normals are computed internally (radius `setRadius`). The response is
// written to the output's `intensity`.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class HarrisKeypoint6D {
public:
    HarrisKeypoint6D(float radius, float threshold);

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setRadius(float radius);
    void setThreshold(float threshold);
    void setNonMaxSupression(bool suppress);
    void setRefine(bool do_refine);
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
