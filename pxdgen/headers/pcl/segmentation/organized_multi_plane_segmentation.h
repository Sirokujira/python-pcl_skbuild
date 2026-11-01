// Mirror header: self-contained stand-in for <pcl/segmentation/organized_multi_plane_segmentation.h>.
//
// Every plane in an ORGANIZED cloud at once, by growing connected
// components of agreeing normals over the pixel grid. The overloads that
// return Eigen centroids / covariances / PlanarRegions are left out.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointNT, typename PointLT>
class OrganizedMultiPlaneSegmentation {
public:
    OrganizedMultiPlaneSegmentation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setMinInliers(unsigned int min_inliers);
    unsigned int getMinInliers() const;
    void setAngularThreshold(double angular_threshold);
    double getAngularThreshold() const;
    void setDistanceThreshold(double distance_threshold);
    double getDistanceThreshold() const;
    void setMaximumCurvature(double maximum_curvature);
    double getMaximumCurvature() const;
    void setProjectPoints(bool project_points);
    void segment(std::vector<ModelCoefficients>& model_coefficients,
                 std::vector<PointIndices>& inlier_indices);
};

}  // namespace pcl
