// Mirror header: self-contained stand-in for <pcl/sample_consensus/sac.h>,
// the SampleConsensus base every method (RANSAC, LMedS, MSAC, RRANSAC,
// RMSAC, MLESAC, PROSAC) derives from. The methods are constructed and
// read out through compat/sac_models.h.
#pragma once

namespace pcl {

template <typename PointT>
class SampleConsensus {
public:
    void setDistanceThreshold(double threshold);
    double getDistanceThreshold() const;
    void setMaxIterations(int max_iterations);
    int getMaxIterations() const;
    void setProbability(double probability);
    double getProbability() const;
    void setNumberOfThreads(int nr_threads);
    int getNumberOfThreads() const;
};

}  // namespace pcl
