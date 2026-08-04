// Mirror header: self-contained stand-in for <pcl/features/vfh.h>.
//
// VFH describes a whole cloud with one 308-bin histogram, so `compute`
// fills an output cloud holding exactly one point.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class VFHEstimation {
public:
    VFHEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setViewPoint(float vpx, float vpy, float vpz);
    void setNormalizeBins(bool normalize);
    void setNormalizeDistance(bool normalize);
    void setFillSizeComponent(bool fill_size);
};

}  // namespace pcl
