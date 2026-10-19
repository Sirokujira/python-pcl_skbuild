// Mirror header: self-contained stand-in for <pcl/features/rsd.h>.
//
// Radius-based Surface Descriptor: the smallest and largest surface
// radius around each point. Radius search only -- setKSearch is a
// no-op that prints an error in PCL, so it is left out.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class RSDEstimation {
public:
    RSDEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNrSubdivisions(int nr_subdiv);
    int getNrSubdivisions() const;
    void setPlaneRadius(double plane_radius);
    double getPlaneRadius() const;
};

}  // namespace pcl
