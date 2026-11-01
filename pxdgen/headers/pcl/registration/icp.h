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

// Point-to-plane ICP on oriented clouds (PCL declares it in this same
// header). The Scalar template argument keeps its default.
template <typename PointSource, typename PointTarget>
class IterativeClosestPointWithNormals {
public:
    IterativeClosestPointWithNormals();

    // Inherited entry points (Registration / IterativeClosestPoint).
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

    void setUseSymmetricObjective(bool use_symmetric_objective);
    bool getUseSymmetricObjective() const;
    void setEnforceSameDirectionNormals(bool enforce_same_direction_normals);
    bool getEnforceSameDirectionNormals() const;
};

}  // namespace pcl
