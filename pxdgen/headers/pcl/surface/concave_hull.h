// Mirror header: self-contained stand-in for
// <pcl/surface/concave_hull.h>.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT>
class ConcaveHull {
public:
    ConcaveHull();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);

    void setAlpha(double alpha);
    double getAlpha() const;
    void setDimension(int dimension);
    int getDimension() const;

    void reconstruct(PointCloud<PointInT>& points);
};

}  // namespace pcl
