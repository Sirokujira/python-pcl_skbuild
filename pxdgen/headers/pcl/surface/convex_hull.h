// Mirror header: self-contained stand-in for
// <pcl/surface/convex_hull.h>.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT>
class ConvexHull {
public:
    ConvexHull();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);

    void setDimension(int dimension);
    int getDimension() const;
    void setComputeAreaVolume(bool value);
    double getTotalArea() const;
    double getTotalVolume() const;

    void reconstruct(PointCloud<PointInT>& points);
    void reconstruct(PointCloud<PointInT>& points,
                     std::vector<Vertices>& polygons);
};

}  // namespace pcl
