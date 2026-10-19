// Mirror header: self-contained stand-in for <pcl/features/grsd.h>.
//
// Global RSD: one 21-bin descriptor for a whole cloud from the
// distribution of its RSD surface types. Needs normals and a radius
// (which doubles as the voxel width).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class GRSDEstimation {
public:
    GRSDEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointOutT>& output);

    void setRadiusSearch(double radius);
    double getRadiusSearch() const;
    void setNrSubdivisions(int nr_subdiv);
    void setPlaneRadius(double plane_radius);
};

}  // namespace pcl
