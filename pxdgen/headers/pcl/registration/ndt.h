// Mirror header: self-contained stand-in for
// <pcl/registration/ndt.h> (Normal Distributions Transform).
//
// strawlab/python-pcl#265 asks for this one by name and it was never
// added there; the repository is archived, so this is the wrapper's to
// provide.
//
// setInputCloud is NDT's own spelling of setInputSource (it predates the
// rename and both exist). See filters/voxel_grid.h for the flattening
// rationale.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointSource, typename PointTarget>
class NormalDistributionsTransform {
public:
    NormalDistributionsTransform();

    // Inherited entry points (Registration).
    void setInputSource(const std::shared_ptr<PointCloud<PointSource>>& cloud);
    void setInputTarget(const std::shared_ptr<PointCloud<PointTarget>>& cloud);
    void setMaximumIterations(int nr_iterations);
    int getMaximumIterations();
    void setTransformationEpsilon(double epsilon);
    double getTransformationEpsilon();
    void setEuclideanFitnessEpsilon(double epsilon);
    void align(PointCloud<PointSource>& output);
    bool hasConverged();
    double getFitnessScore();

    void setResolution(float resolution);
    float getResolution() const;
    void setStepSize(double step_size);
    double getStepSize() const;
    void setOulierRatio(double outlier_ratio);
    double getOulierRatio() const;
    double getTransformationProbability() const;
    int getFinalNumIteration() const;
};

}  // namespace pcl
