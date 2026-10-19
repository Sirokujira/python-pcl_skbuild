// Mirror header: self-contained stand-in for <pcl/features/gasd.h>.
//
// Globally Aligned Spatial Distribution: one descriptor for a whole
// cloud after aligning it to its own reference frame; the colour
// variant adds colour histograms. setViewDirection takes an Eigen
// vector and is left out (the default view is +z).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

enum HistogramInterpolationMethod {
    INTERP_NONE,
    INTERP_TRILINEAR,
    INTERP_QUADRILINEAR
};

template <typename PointInT, typename PointOutT>
class GASDEstimation {
public:
    GASDEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setShapeHalfGridSize(std::size_t shgs);
    void setShapeHistsSize(std::size_t shs);
    void setShapeHistsInterpMethod(HistogramInterpolationMethod interp);
};

template <typename PointInT, typename PointOutT>
class GASDColorEstimation {
public:
    GASDColorEstimation();

    // Inherited entry points (PCLBase / Feature / GASDEstimation).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);
    void setShapeHalfGridSize(std::size_t shgs);
    void setShapeHistsSize(std::size_t shs);
    void setShapeHistsInterpMethod(HistogramInterpolationMethod interp);

    void setColorHalfGridSize(std::size_t chgs);
    void setColorHistsSize(std::size_t chs);
    void setColorHistsInterpMethod(HistogramInterpolationMethod interp);
};

}  // namespace pcl
