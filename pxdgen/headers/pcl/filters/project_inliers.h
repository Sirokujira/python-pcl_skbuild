// Mirror header: self-contained stand-in for
// <pcl/filters/project_inliers.h>.
//
// Projects points onto a parametric model — the usual next step after
// SACSegmentation, using the coefficients it returned.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ProjectInliers {
public:
    ProjectInliers();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setModelType(int model);
    int getModelType();
    void setModelCoefficients(
        const std::shared_ptr<ModelCoefficients>& model);
    void setCopyAllData(bool val);
    bool getCopyAllData();
};

}  // namespace pcl
