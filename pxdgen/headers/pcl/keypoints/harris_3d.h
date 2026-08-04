// Mirror header: self-contained stand-in for
// <pcl/keypoints/harris_3d.h>.
//
// The Harris response lands in the output point's intensity field, which
// is why PointOutT is PointXYZI rather than the input type.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class HarrisKeypoint3D {
public:
    enum ResponseMethod { HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE };

    HarrisKeypoint3D();
    HarrisKeypoint3D(ResponseMethod method, float radius, float threshold);

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setMethod(ResponseMethod type);
    void setRadius(float radius);
    void setThreshold(float threshold);
    void setNonMaxSupression(bool suppress);
    void setRefine(bool do_refine);
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
