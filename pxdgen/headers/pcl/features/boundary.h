// Mirror header: self-contained stand-in for <pcl/features/boundary.h>.
//
// Flags the points on the edge of a surface: each point's neighbours are
// projected onto its tangent plane, and a gap wider than the angle
// threshold between consecutive neighbours marks a boundary. Needs
// normals (setInputNormals).
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class BoundaryEstimation {
public:
    BoundaryEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    int getKSearch() const;
    void setRadiusSearch(double radius);
    double getRadiusSearch() const;
    void compute(PointCloud<PointOutT>& output);

    void setAngleThreshold(float angle);
    float getAngleThreshold();
};

}  // namespace pcl
