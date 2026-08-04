// Mirror header: self-contained stand-in for the
// SACSegmentationFromNormals half of <pcl/segmentation/sac_segmentation.h>.
//
// Its own file rather than an addition to sac_segmentation.h because a
// mirror header maps one-to-one onto one generated pxd, and this class
// pulls in the Normal point type the plain segmenter does not need. Both
// name the same real header through `extern_from`.
//
// setAxis takes an Eigen::Vector3f, which a mirror header may not name —
// pcl/compat/eigen_args.h takes plain floats.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointNT>
class SACSegmentationFromNormals {
public:
    SACSegmentationFromNormals();

    // Inherited entry points (PCLBase / SACSegmentation).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setModelType(int model);
    void setMethodType(int method);
    void setDistanceThreshold(double threshold);
    void setMaxIterations(int max_iterations);
    void setOptimizeCoefficients(bool optimize);
    void setProbability(double probability);
    void setRadiusLimits(const double& min_radius, const double& max_radius);
    void setEpsAngle(double ea);
    double getEpsAngle() const;
    void segment(PointIndices& inliers, ModelCoefficients& model_coefficients);

    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setNormalDistanceWeight(double distance_weight);
    double getNormalDistanceWeight() const;
    void setMinMaxOpeningAngle(const double& min_angle,
                               const double& max_angle);
    void setDistanceFromOrigin(const double d);
    double getDistanceFromOrigin() const;
};

}  // namespace pcl
