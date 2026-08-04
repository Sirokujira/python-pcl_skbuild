// Mirror header: self-contained stand-in for
// <pcl/registration/icp.h>, plus the two variants that share its API
// exactly (icp_nl.h, gicp.h get their own headers).
//
// getFinalTransformation returns an Eigen::Matrix4f, which a mirror
// header may not name — it is reached through
// pcl/compat/registration_result.h, which copies the 16 floats out.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointSource, typename PointTarget>
class IterativeClosestPoint {
public:
    IterativeClosestPoint();

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

    void setUseReciprocalCorrespondences(bool use_reciprocal_correspondence);
    bool getUseReciprocalCorrespondences() const;
};

}  // namespace pcl
