// Mirror header: self-contained stand-in for <pcl/features/our_cvfh.h>.
//
// Oriented, Unique and Repeatable CVFH: CVFH plus a repeatable
// reference frame per region, so the descriptor is pose-aware. The
// Eigen-returning accessors (transforms, centroids) are left out.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class OURCVFHEstimation {
public:
    OURCVFHEstimation();

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
    void setRefineClusters(float rc);
    void setAxisRatio(float f);
    void setMinAxisValue(float f);
};

}  // namespace pcl
