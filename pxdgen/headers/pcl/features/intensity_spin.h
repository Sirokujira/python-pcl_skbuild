// Mirror header: self-contained stand-in for
// <pcl/features/intensity_spin.h>.
//
// Intensity spin images (Lazebnik et al.): a 2-D histogram of distance
// from the point against intensity. The output type is fixed at compile
// time, so the bin counts' product must match it (enforced above).
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class IntensitySpinEstimation {
public:
    IntensitySpinEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNrDistanceBins(std::size_t nr_distance_bins);
    int getNrDistanceBins();
    void setNrIntensityBins(std::size_t nr_intensity_bins);
    int getNrIntensityBins();
    void setSmoothingBandwith(float sigma);
    float getSmoothingBandwith();
};

}  // namespace pcl
