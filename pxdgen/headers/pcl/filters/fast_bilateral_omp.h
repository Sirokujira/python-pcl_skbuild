// Mirror header: self-contained stand-in for
// <pcl/filters/fast_bilateral_omp.h>.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class FastBilateralFilterOMP {
public:
    explicit FastBilateralFilterOMP(unsigned int nr_threads);

    // Inherited entry points (PCLBase / Filter / FastBilateralFilter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setSigmaS(float sigma_s);
    float getSigmaS() const;
    void setSigmaR(float sigma_r);
    float getSigmaR() const;

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
