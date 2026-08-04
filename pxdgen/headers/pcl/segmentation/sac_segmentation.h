// Mirror header: self-contained stand-in for
// <pcl/segmentation/sac_segmentation.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class SACSegmentation {
public:
    SACSegmentation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setModelType(int model);
    int getModelType() const;
    void setMethodType(int method);
    int getMethodType() const;
    void setDistanceThreshold(double threshold);
    double getDistanceThreshold() const;
    void setMaxIterations(int max_iterations);
    int getMaxIterations() const;
    void setProbability(double probability);
    double getProbability() const;
    void setOptimizeCoefficients(bool optimize);
    bool getOptimizeCoefficients() const;
    void setRadiusLimits(const double& min_radius, const double& max_radius);
    void setEpsAngle(double ea);
    double getEpsAngle() const;

    void segment(PointIndices& inliers, ModelCoefficients& model_coefficients);
};

}  // namespace pcl
