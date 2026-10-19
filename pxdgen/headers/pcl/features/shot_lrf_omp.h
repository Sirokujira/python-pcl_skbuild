// Mirror header: self-contained stand-in for <pcl/features/shot_lrf_omp.h>.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class SHOTLocalReferenceFrameEstimationOMP {
public:
    SHOTLocalReferenceFrameEstimationOMP();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
