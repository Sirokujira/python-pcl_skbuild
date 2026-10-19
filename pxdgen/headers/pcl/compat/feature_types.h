// Mirror header for src/pcl/compat/feature_types.h.
//
// Three things a pxd cannot spell: pcl::Histogram<N> (an integer
// template argument), SUSANKeypoint's fourth template argument (an
// intensity accessor type) and CRHEstimation::setCentroid (an Eigen
// vector). Each gets ONE plain name here.
#pragma once

#include <memory>

#include <pcl/features/crh.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// pcl::Histogram<153>: a spin image of width 8. Declared with `class`
// so the pxd gets a cppclass and Cython spells the bare name: the real
// name is a typedef, and `struct pclcompat::Histogram153` (what a cdef
// struct would produce) is ill-formed for a typedef.
class Histogram153 {
public:
    float histogram[153];
};

// pcl::Histogram<90>: a camera roll histogram (same reasoning).
class Histogram90 {
public:
    float histogram[90];
};

// pcl::SUSANKeypoint<pcl::PointXYZRGB, pcl::PointXYZ> with its default
// normal type and intensity accessor.
class SUSANKeypointRGB {
public:
    SUSANKeypointRGB();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>& cloud);
    void compute(pcl::PointCloud<pcl::PointXYZ>& output);

    void setRadius(float radius);
    void setDistanceThreshold(float distance_threshold);
    void setAngularThreshold(float angular_threshold);
    void setIntensityThreshold(float intensity_threshold);
    void setNormals(const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals);
    void setNumberOfThreads(unsigned int nr_threads);
    void setNonMaxSupression(bool nonmax);
    void setGeometricValidation(bool validate);
};

void setCRHCentroid(
    pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, Histogram90>& estimator,
    float x, float y, float z);

}  // namespace pclcompat
