// Mirror header: self-contained stand-in for
// <pcl/features/moment_of_inertia_estimation.h>.
//
// The Eigen-typed getters (getEigenVectors, getMassCenter, getOBB, ...)
// are reached through pcl/compat/moment_of_inertia.h, which flattens
// their Eigen vectors into plain float buffers. Only the scalar and
// std::vector parts of the API are declared here.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class MomentOfInertiaEstimation {
public:
    MomentOfInertiaEstimation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void compute();
    void setAngleStep(float step);
    float getAngleStep() const;
    void setNormalizePointMassFlag(bool need_to_normalize);
    bool getNormalizePointMassFlag() const;
    void setPointMass(float point_mass);
    float getPointMass() const;

    bool getMomentOfInertia(std::vector<float>& moment_of_inertia) const;
    bool getEccentricity(std::vector<float>& eccentricity) const;
};

}  // namespace pcl
