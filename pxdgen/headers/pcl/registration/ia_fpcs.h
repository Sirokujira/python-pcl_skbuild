// Mirror header: self-contained stand-in for <pcl/registration/ia_fpcs.h>.
//
// 4-Points Congruent Sets: a global (no initial guess) registration that
// matches coplanar 4-point bases between the clouds. The NormalT and
// Scalar template arguments keep their defaults.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {
namespace registration {

template <typename PointSource, typename PointTarget>
class FPCSInitialAlignment {
public:
    FPCSInitialAlignment();

    // Inherited entry points (Registration).
    void setInputSource(const std::shared_ptr<PointCloud<PointSource>>& cloud);
    void setInputTarget(const std::shared_ptr<PointCloud<PointTarget>>& cloud);
    void setMaximumIterations(int nr_iterations);
    void setMaxCorrespondenceDistance(double distance_threshold);
    void align(PointCloud<PointSource>& output);
    bool hasConverged();
    double getFitnessScore();

    void setDelta(float delta, bool normalize);
    float getDelta() const;
    void setApproxOverlap(float approx_overlap);
    float getApproxOverlap() const;
    void setScoreThreshold(float score_threshold);
    float getScoreThreshold() const;
    void setNumberOfSamples(int nr_samples);
    void setMaxNormalDifference(float max_norm_diff);
    void setMaxComputationTime(int max_runtime);
    void setNumberOfThreads(int nr_threads);
};

}  // namespace registration
}  // namespace pcl
