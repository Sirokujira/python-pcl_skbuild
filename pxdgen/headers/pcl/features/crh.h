// Mirror header: self-contained stand-in for <pcl/features/crh.h>.
//
// Camera Roll Histogram: a 90-bin histogram of the cloud's roll about
// the viewing direction, used with CVFH to recover the one degree of
// freedom it leaves open. setCentroid takes an Eigen vector and goes
// through compat/feature_types.h; the output type is Histogram<90>.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class CRHEstimation {
public:
    CRHEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setViewPoint(float vpx, float vpy, float vpz);
};

}  // namespace pcl
