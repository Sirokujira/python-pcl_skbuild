// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/outofcore_args.h (reached as
// "pcl/compat/outofcore_args.h" because CMake adds src/ to the
// includes). The real header needs VTK; this one names only what
// Cython has to see.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class OutofcoreTree {
public:
    OutofcoreTree();
    void create(const std::string& path, const double* min3, const double* max3,
                unsigned int depth);
    void open(const std::string& path);
    std::uint64_t add(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);
    void buildLOD();
    void query(const double* min3, const double* max3, unsigned int depth,
               pcl::PointCloud<pcl::PointXYZ>& out) const;
    void voxelCentres(unsigned int depth, std::vector<double>& out) const;
    unsigned int depth() const;
    double voxelSideLength(unsigned int depth) const;
    std::uint64_t pointsAtDepth(unsigned int depth) const;
    void boundingBox(double* min3, double* max3) const;
};

}  // namespace pclcompat
