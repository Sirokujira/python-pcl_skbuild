// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/tracking_args.h (reached as
// "pcl/compat/tracking_args.h" because CMake adds src/ to the includes).
//
// The real header wires up PCL's tracker, coherence chain and search
// method and names Eigen types throughout; this one names only the class
// Cython has to see, so pxd generation still runs on a machine with
// neither PCL nor Eigen installed.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class ParticleTracker {
public:
    ParticleTracker(int particle_num, double step_noise, double resolution,
                    double maximum_distance, int threads);

    void setReferenceCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& reference);
    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);

    void compute();

    void result(float* out7);
    void resultTransform(float* out16);
    void alignedReference(std::vector<pcl::PointXYZ>& out);
    int particleNum();
};

}  // namespace pclcompat
