// Mirror header for src/pcl/compat/feature_types.h.
//
// Three things a pxd cannot spell: pcl::Histogram<N> (an integer
// template argument), SUSANKeypoint's fourth template argument (an
// intensity accessor type) and CRHEstimation::setCentroid (an Eigen
// vector). Each gets ONE plain name here.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
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

class Histogram20 {
public:
    float histogram[20];
};

class Histogram32 {
public:
    float histogram[32];
};

class Histogram135 {
public:
    float histogram[135];
};

// OrganizedEdgeBase's EDGELABEL_* bits.
const int EDGELABEL_NAN_BOUNDARY = 1;
const int EDGELABEL_OCCLUDING = 2;
const int EDGELABEL_OCCLUDED = 4;
const int EDGELABEL_HIGH_CURVATURE = 8;
const int EDGELABEL_RGB_CANNY = 16;

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

// pcl::computeApproximateNormals: per-vertex normals of a mesh, averaged
// from the faces around each vertex.
void approximateNormals(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                        const std::vector<pcl::Vertices>& polygons,
                        pcl::PointCloud<pcl::Normal>& normals);

// pcl::computePairFeatures on xyz / normal triples: the four PFH pair
// features (f1..f3 angles, f4 distance). False for a degenerate pair.
bool pairFeatures(const float* p1, const float* n1, const float* p2,
                  const float* n2, float* out4);

// StatisticalMultiscaleInterestRegionExtraction: one index list per
// region of interest.
void multiscaleInterestRegions(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::vector<float>& scales,
    std::vector<std::vector<int>>& out_regions);

// MultiscaleFeaturePersistence over FPFH: the features (and the input
// indices they belong to) that stay distinctive across all `scales`.
void persistentFPFH(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    const std::vector<float>& scales, float alpha, int norm_type,
    pcl::PointCloud<pcl::FPFHSignature33>& out_features,
    std::vector<int>& out_indices);

// pcl::NormType values.
const int NORM_L1 = 0;
const int NORM_L2_SQR = 1;
const int NORM_L2 = 2;
const int NORM_LINF = 3;
const int NORM_JM = 4;
const int NORM_B = 5;
const int NORM_SUBLINEAR = 6;
const int NORM_CS = 7;
const int NORM_DIV = 8;
const int NORM_PF = 9;
const int NORM_K = 10;
const int NORM_KL = 11;
const int NORM_HIK = 12;

}  // namespace pclcompat
