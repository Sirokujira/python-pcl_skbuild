// Mirror header: self-contained stand-in for <pcl/registration/ia_kfpcs.h>.
//
// Keypoint-based 4PCS: FPCS over keypoints with translation bounds and a
// candidate list; the FPCS settings are flattened in.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {
namespace registration {

template <typename PointSource, typename PointTarget>
class KFPCSInitialAlignment {
public:
    KFPCSInitialAlignment();

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

    void setUpperTranslationThreshold(float upper_trl_boundary);
    float getUpperTranslationThreshold() const;
    void setLowerTranslationThreshold(float lower_trl_boundary);
    float getLowerTranslationThreshold() const;
    void setLambda(float lambda);
    float getLambda() const;
};

}  // namespace registration
}  // namespace pcl
