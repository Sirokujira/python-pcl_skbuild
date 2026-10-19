// Mirror header: self-contained stand-in for
// <pcl/keypoints/smoothed_surfaces_keypoint.h>.
//
// Keypoints that persist across several smoothed versions of a surface
// (Rusu et al.'s scale-space keypoints). Each smoothed cloud is added
// with its normals, a search tree and its scale; the tree is built in
// pcl/compat/keypoint_args.h.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointNT>
class SmoothedSurfacesKeypoint {
public:
    SmoothedSurfacesKeypoint();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void compute(PointCloud<PointT>& output);

    void resetClouds();
    void setNeighborhoodConstant(float neighborhood_constant);
    float getNeighborhoodConstant();
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setInputScale(float input_scale);
};

}  // namespace pcl
