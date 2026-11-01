// Mirror header: self-contained stand-in for <pcl/features/shot_omp.h>.
//
// The OpenMP variants of SHOTEstimation and SHOTColorEstimation. The
// reference-frame type parameter is left to its default.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class SHOTEstimationOMP {
public:
    explicit SHOTEstimationOMP(unsigned int nr_threads);

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);
    void setLRFRadius(float radius);

    void setNumberOfThreads(unsigned int nr_threads);
};

template <typename PointInT, typename PointNT, typename PointOutT>
class SHOTColorEstimationOMP {
public:
    SHOTColorEstimationOMP(bool describe_shape, bool describe_color,
                           unsigned int nr_threads);

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);
    void setLRFRadius(float radius);

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
