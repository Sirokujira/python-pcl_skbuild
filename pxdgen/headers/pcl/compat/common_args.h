// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/common_args.h (reached as "pcl/compat/common_args.h"
// because CMake adds src/ to the includes).
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class PrincipalComponents {
public:
    PrincipalComponents();
    void setInputCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);
    void mean(float* out3);
    void eigenVectors(float* out9);
    void eigenValues(float* out3);
    void project(const pcl::PointCloud<pcl::PointXYZ>& in,
                 pcl::PointCloud<pcl::PointXYZ>& out);
    void reconstruct(const pcl::PointCloud<pcl::PointXYZ>& in,
                     pcl::PointCloud<pcl::PointXYZ>& out);
};

bool lineWithLine(const float* a6, const float* b6, float* point3, double sqr_eps);
bool planeWithPlane(const float* a4, const float* b4, float* line6,
                    double angular_tolerance);
bool threePlanes(const float* a4, const float* b4, const float* c4, float* point3,
                 double determinant_tolerance);

float histogramNorm(const std::vector<float>& a, const std::vector<float>& b,
                    int type, float p1, float p2);

void transformationFromCorrespondences(const pcl::PointCloud<pcl::PointXYZ>& from,
                                       const pcl::PointCloud<pcl::PointXYZ>& to,
                                       const std::vector<float>& weights,
                                       float* out16);

double estimateProjectionMatrix(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    float* projection12, float* camera9);

}  // namespace pclcompat
