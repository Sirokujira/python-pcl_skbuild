// Mirror header: self-contained stand-in for <pcl/features/principal_curvatures.h>.
//
// Principal curvatures from the normals of each point's neighbourhood.
// Needs normals and a neighbourhood size.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class PrincipalCurvaturesEstimation {
public:
    PrincipalCurvaturesEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
