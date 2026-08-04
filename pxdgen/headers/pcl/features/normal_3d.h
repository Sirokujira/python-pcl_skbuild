// Mirror header: self-contained stand-in for <pcl/features/normal_3d.h>.
//
// setSearchMethod takes a search::Search<PointT>::Ptr, which would drag
// PCL's whole search hierarchy in; NormalEstimation builds a default
// KdTree when none is set, so it is left out rather than half-declared.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class NormalEstimation {
public:
    NormalEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setKSearch(int k);
    int getKSearch() const;
    void setRadiusSearch(double radius);
    double getRadiusSearch() const;
    void compute(PointCloud<PointOutT>& output);

    void setViewPoint(float vpx, float vpy, float vpz);
    void useSensorOriginAsViewPoint();
};

}  // namespace pcl
