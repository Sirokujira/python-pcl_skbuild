// Mirror header: self-contained stand-in for
// <pcl/filters/radius_outlier_removal.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class RadiusOutlierRemoval {
public:
    RadiusOutlierRemoval();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);
    bool getNegative() const;
    void setKeepOrganized(bool keep_organized);

    void setRadiusSearch(double radius);
    double getRadiusSearch();
    void setMinNeighborsInRadius(int min_pts);
    int getMinNeighborsInRadius();
};

}  // namespace pcl
