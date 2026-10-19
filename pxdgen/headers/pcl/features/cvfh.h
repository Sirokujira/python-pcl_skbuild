// Mirror header: self-contained stand-in for <pcl/features/cvfh.h>.
//
// Clustered Viewpoint Feature Histogram: VFH computed once per smooth
// region of the cloud (regions are grown over points whose normals agree
// within the EPS angle and whose curvature is below the threshold), so a
// partial view yields several 308-bin descriptors instead of one.
// getCentroidClusters hands back Eigen vectors and is left out.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class CVFHEstimation {
public:
    CVFHEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setViewPoint(float vpx, float vpy, float vpz);
    void setRadiusNormals(float radius_normals);
    void setClusterTolerance(float d);
    void setEPSAngleThreshold(float d);
    void setCurvatureThreshold(float d);
    void setMinPoints(std::size_t min);
    void setNormalizeBins(bool normalize);
};

}  // namespace pcl
