// Mirror header: self-contained stand-in for <pcl/filters/crop_hull.h>.
//
// Keeps the points inside (or outside) a hull, which is what makes
// ConvexHull/ConcaveHull useful as a region selector rather than just a
// shape.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class CropHull {
public:
    CropHull();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setHullCloud(std::shared_ptr<PointCloud<PointT>> points);
    void setHullIndices(const std::vector<Vertices>& polygons);
    void setDim(int dim);
    void setCropOutside(bool crop_outside);
};

}  // namespace pcl
