// Mirror header: self-contained stand-in for
// <pcl/registration/gicp.h>.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointSource, typename PointTarget>
class GeneralizedIterativeClosestPoint {
public:
    GeneralizedIterativeClosestPoint();

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

    void setRotationEpsilon(double epsilon);
    double getRotationEpsilon();
    void setCorrespondenceRandomness(int k);
    int getCorrespondenceRandomness();
    void setMaximumOptimizerIterations(int max_iterations);
    int getMaximumOptimizerIterations();
};

}  // namespace pcl
