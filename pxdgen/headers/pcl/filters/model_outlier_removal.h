// Mirror header: self-contained stand-in for <pcl/filters/model_outlier_removal.h>.
//
// Keeps the points within a distance of a given SAC model (plane, sphere,
// ...) whose coefficients are already known. setInputNormals takes the
// ConstPtr by value in PCL; declaring the non-const Ptr converts.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>

namespace pcl {

template <typename PointT>
class ModelOutlierRemoval {
public:
    ModelOutlierRemoval();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setModelCoefficients(const ModelCoefficients& model_coefficients);
    void setModelType(SacModel model);
    SacModel getModelType() const;
    void setThreshold(float thresh);
    float getThreshold() const;
    void setInputNormals(const std::shared_ptr<PointCloud<Normal>>& normals_ptr);
    void setNormalDistanceWeight(double weight);
    double getNormalDistanceWeight() const;
};

}  // namespace pcl
