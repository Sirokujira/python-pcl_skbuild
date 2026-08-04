// Mirror header: self-contained stand-in for
// <pcl/filters/statistical_outlier_removal.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class StatisticalOutlierRemoval {
public:
    StatisticalOutlierRemoval();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);
    bool getNegative() const;
    void setKeepOrganized(bool keep_organized);

    void setMeanK(int nr_k);
    int getMeanK();
    void setStddevMulThresh(double stddev_mult);
    double getStddevMulThresh();
};

}  // namespace pcl
