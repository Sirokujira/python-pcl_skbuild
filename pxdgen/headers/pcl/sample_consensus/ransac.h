// Mirror header: self-contained stand-in for
// <pcl/sample_consensus/ransac.h>.
//
// Construction, computeModel, getInliers and getModelCoefficients all go
// through pcl/compat/sac_models.h — the constructor needs a
// derived-to-base shared_ptr conversion and the coefficients come back in
// an Eigen::VectorXf. What is left here is the plain configuration.
#pragma once

namespace pcl {

template <typename PointT>
class RandomSampleConsensus {
public:
    void setDistanceThreshold(double threshold);
    double getDistanceThreshold();
    void setMaxIterations(int max_iterations);
    int getMaxIterations();
    void setProbability(double probability);
    double getProbability();
};

}  // namespace pcl
