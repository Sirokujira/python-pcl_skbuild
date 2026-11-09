// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/normal_signature.h (reached as
// "pcl/compat/normal_signature.h" because CMake adds src/ to the
// includes). PCL's NormalBasedSignatureEstimation with computeFeature
// corrected (it returns only NaN); the base-class setters it needs are
// flattened here.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class NormalSignature {
public:
    NormalSignature();
    void setInputCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);
    void setInputNormals(const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals);
    void setRadiusSearch(double radius);
    void setN(std::size_t n);
    std::size_t getN();
    void setM(std::size_t m);
    std::size_t getM();
    void setScale(float scale);
    float getScale();
    void computeAt(const std::vector<int>& indices, std::vector<float>& out);
};

}  // namespace pclcompat
