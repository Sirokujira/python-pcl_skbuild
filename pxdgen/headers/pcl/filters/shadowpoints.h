// Mirror header: self-contained stand-in for <pcl/filters/shadowpoints.h>.
//
// Drops points whose normal is nearly perpendicular to the line of sight
// (seen edge-on, so their position is unreliable). Needs normals.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename NormalT>
class ShadowPoints {
public:
    ShadowPoints();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setNormals(const std::shared_ptr<PointCloud<NormalT>>& normals);
    void setThreshold(float threshold);
    float getThreshold() const;
};

}  // namespace pcl
