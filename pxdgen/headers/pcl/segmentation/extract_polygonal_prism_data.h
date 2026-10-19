// Mirror header: self-contained stand-in for <pcl/segmentation/extract_polygonal_prism_data.h>.
//
// The points inside a prism: above a planar polygon (a hull cloud) and
// within a height range. The out-parameter getters are left out.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ExtractPolygonalPrismData {
public:
    ExtractPolygonalPrismData();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setInputPlanarHull(const std::shared_ptr<PointCloud<PointT>>& hull);
    void setHeightLimits(double height_min, double height_max);
    void setViewPoint(float vpx, float vpy, float vpz);
    void segment(PointIndices& output);
};

}  // namespace pcl
