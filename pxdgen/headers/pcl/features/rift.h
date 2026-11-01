// Mirror header: self-contained stand-in for <pcl/features/rift.h>.
//
// Rotation Invariant Feature Transform (Lazebnik et al.): histograms of
// intensity-gradient orientation over concentric rings. Takes the
// gradients IntensityGradientEstimation computes.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename GradientT, typename PointOutT>
class RIFTEstimation {
public:
    RIFTEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setInputGradient(const std::shared_ptr<PointCloud<GradientT>>& gradient);
    void setNrDistanceBins(int nr_distance_bins);
    int getNrDistanceBins() const;
    void setNrGradientBins(int nr_gradient_bins);
    int getNrGradientBins() const;
};

}  // namespace pcl
