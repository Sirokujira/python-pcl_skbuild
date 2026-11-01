// Mirror header: self-contained stand-in for <pcl/features/fpfh_omp.h>.
//
// FPFHEstimation with OpenMP over the input points; same output.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class FPFHEstimationOMP {
public:
    explicit FPFHEstimationOMP(unsigned int nr_threads);

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
