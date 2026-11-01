// Mirror header: self-contained stand-in for <pcl/keypoints/trajkovic_3d.h>.
//
// Trajkovic corners on the NORMALS of an organized cloud: the 3-D
// analogue of the 2-D detector. The normals type parameter is left to
// its default (pcl::Normal).
#pragma once

#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class TrajkovicKeypoint3D {
public:
    TrajkovicKeypoint3D();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setWindowSize(int window_size);
    int getWindowSize() const;
    void setFirstThreshold(float threshold);
    float getFirstThreshold() const;
    void setSecondThreshold(float threshold);
    float getSecondThreshold() const;
    void setNormals(const std::shared_ptr<PointCloud<Normal>>& normals);
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
