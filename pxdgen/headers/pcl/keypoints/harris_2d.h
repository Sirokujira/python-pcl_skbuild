// Mirror header: self-contained stand-in for <pcl/keypoints/harris_2d.h>.
//
// Harris corners on the intensity image of an organized cloud. The
// ResponseMethod enum goes through pcl/compat/keypoint_args.h; the
// intensity-selector parameter is left to its default.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class HarrisKeypoint2D {
public:
    HarrisKeypoint2D();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);
    void setKSearch(int k);

    void setWindowWidth(int window_width);
    void setWindowHeight(int window_height);
    void setSkippedPixels(int skipped_pixels);
    void setMinimalDistance(int min_distance);
    void setThreshold(float threshold);
    void setNonMaxSupression(bool suppress);
    void setRefine(bool do_refine);
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
