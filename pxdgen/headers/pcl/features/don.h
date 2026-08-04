// Mirror header: self-contained stand-in for <pcl/features/don.h>
// (difference of normals).
//
// Scale-based segmentation: the difference between normals estimated at
// two radii is large exactly where the surface has structure at that
// scale, so thresholding it isolates features of a chosen size.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class DifferenceOfNormalsEstimation {
public:
    DifferenceOfNormalsEstimation();

    // Inherited entry point (PCLBase). NOT `compute`: DoN makes that
    // one private and exposes computeFeature instead, because the
    // difference is only defined once both scales are set.
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void computeFeature(PointCloud<PointOutT>& output);

    void setNormalScaleSmall(
        const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setNormalScaleLarge(
        const std::shared_ptr<PointCloud<PointNT>>& normals);
    bool initCompute();
};

}  // namespace pcl
