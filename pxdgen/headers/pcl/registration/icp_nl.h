// Mirror header: self-contained stand-in for
// <pcl/registration/icp_nl.h>.
//
// IterativeClosestPointNonLinear adds no API of its own -- it swaps the
// transformation estimator for a Levenberg-Marquardt one -- so what is
// declared here is the inherited Registration surface. See
// filters/voxel_grid.h for the flattening rationale.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointSource, typename PointTarget>
class IterativeClosestPointNonLinear {
public:
    IterativeClosestPointNonLinear();

    // Inherited entry points (Registration).
    void setInputSource(const std::shared_ptr<PointCloud<PointSource>>& cloud);
    void setInputTarget(const std::shared_ptr<PointCloud<PointTarget>>& cloud);
    void setMaximumIterations(int nr_iterations);
    int getMaximumIterations();
    void setMaxCorrespondenceDistance(double distance_threshold);
    double getMaxCorrespondenceDistance();
    void setTransformationEpsilon(double epsilon);
    double getTransformationEpsilon();
    void setEuclideanFitnessEpsilon(double epsilon);
    double getEuclideanFitnessEpsilon();
    void align(PointCloud<PointSource>& output);
    bool hasConverged();
    double getFitnessScore();
};

}  // namespace pcl
