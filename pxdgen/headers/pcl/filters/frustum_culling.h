// Mirror header: self-contained stand-in for <pcl/filters/frustum_culling.h>.
//
// Keeps the points inside a camera frustum. setCameraPose takes an
// Eigen::Matrix4f and goes through the compat shim (eigen_args.h); the
// two-bound FOV getters return through references and are left out.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class FrustumCulling {
public:
    FrustumCulling();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setHorizontalFOV(float hfov);
    void setHorizontalFOV(float fov_left_bound, float fov_right_bound);
    void setVerticalFOV(float vfov);
    void setVerticalFOV(float fov_lower_bound, float fov_upper_bound);
    void setNearPlaneDistance(float np_dist);
    float getNearPlaneDistance() const;
    void setFarPlaneDistance(float fp_dist);
    float getFarPlaneDistance() const;
};

}  // namespace pcl
