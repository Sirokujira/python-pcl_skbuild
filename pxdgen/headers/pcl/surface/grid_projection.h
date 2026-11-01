// Mirror header: self-contained stand-in for
// <pcl/surface/grid_projection.h>.
//
// Li et al.'s grid projection: a mesh with its own vertices, made by
// projecting the corners of the cells near the surface onto it. The
// SurfaceReconstruction base is flattened in.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointNT>
class GridProjection {
public:
    GridProjection();

    // Inherited entry points (PCLBase / SurfaceReconstruction).
    void setInputCloud(const std::shared_ptr<PointCloud<PointNT>>& cloud);
    void reconstruct(PointCloud<PointNT>& points,
                     std::vector<Vertices>& polygons);

    void setResolution(double resolution);
    double getResolution() const;
    void setPaddingSize(int padding_size);
    int getPaddingSize() const;
    void setNearestNeighborNum(int k);
    int getNearestNeighborNum() const;
    void setMaxBinarySearchLevel(int max_binary_search_level);
    int getMaxBinarySearchLevel() const;
};

}  // namespace pcl
