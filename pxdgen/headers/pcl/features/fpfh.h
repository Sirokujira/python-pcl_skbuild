// Mirror header: self-contained stand-in for <pcl/features/fpfh.h>.
//
// FPFH is the per-point descriptor of PCL's registration and recognition
// tutorials: 3 angular features x 11 bins per point, cheap enough to run
// on every point of a scene. Feeding two clouds' descriptors into a
// matcher yields the (model, scene) correspondences that
// GeometricConsistencyGrouping / Hough3DGrouping turn into poses.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class FPFHEstimation {
public:
    FPFHEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    int getKSearch();
    void setRadiusSearch(double radius);
    double getRadiusSearch();
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
