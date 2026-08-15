// Mirror header: self-contained stand-in for <pcl/features/shot.h>.
//
// SHOT is the descriptor PCL's correspondence-grouping tutorial matches:
// 352 bins plus the local reference frame they were computed in. Unlike
// FPFH it REQUIRES a radius — its initCompute() rejects a k-nearest
// setup — so the wrapper never exposes setKSearch.
//
// The real class template has a fourth parameter (PointRFT) defaulting
// to pcl::ReferenceFrame; instantiating with three arguments compiles
// against the real header, so the mirror only declares those three.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class SHOTEstimation {
public:
    SHOTEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    double getRadiusSearch();
    void compute(PointCloud<PointOutT>& output);

    // FeatureWithLocalReferenceFrames: the LRF estimator's own support
    // radius, when it should differ from the descriptor's.
    void setLRFRadius(float radius);
    float getLRFRadius();
};

}  // namespace pcl
