// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/tracking_args.h (reached as
// "pcl/compat/tracking_args.h" because CMake adds src/ to the includes).
//
// The real header wires up PCL's tracker, coherence chain and search
// method and names Eigen types throughout; this one names only what
// Cython has to see, so pxd generation still runs on a machine with
// neither PCL nor Eigen installed.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// The OMP particle filter (plain or KLD-adaptive) over PointT, with a
// distance coherence and whatever addHSVColorCoherence /
// addNormalCoherence add.
template <typename PointT>
class ParticleTrackerT {
public:
    ParticleTrackerT(int particle_num, double step_noise, double resolution,
                     double maximum_distance, int threads);
    ParticleTrackerT(int particle_num, double step_noise, double resolution,
                     double maximum_distance, int threads,
                     int max_particle_num, float bin_size, double epsilon,
                     double delta);
    void setExactCoherence(bool exact);
    void setReferenceCloud(const std::shared_ptr<pcl::PointCloud<PointT>>& reference);
    void setInputCloud(const std::shared_ptr<pcl::PointCloud<PointT>>& cloud);
    void compute();
    void result(float* out7);
    void resultTransform(float* out16);
    void alignedReferenceCloud(pcl::PointCloud<PointT>& out);
    int particleNum();
};

void addHSVColorCoherence(ParticleTrackerT<pcl::PointXYZRGBA>& tracker,
                          double h, double s, double v, double weight);
void addNormalCoherence(ParticleTrackerT<pcl::PointNormal>& tracker,
                        double weight);

// PyramidalKLTTracker over organized PointXYZRGBA frames.
class KLTTracker {
public:
    KLTTracker(int levels, int window_width, int window_height);
    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud);
    void setPointsToTrack(const std::vector<float>& uv);
    void setNumberOfKeypoints(std::size_t n);
    void setAccuracy(float accuracy);
    void setEpsilon(float epsilon);
    void setMaxIterationsNumber(unsigned int n);
    void setNumberOfThreads(unsigned int n);
    void compute();
    bool initialized() const;
    void trackedPoints(std::vector<float>& uv, std::vector<int>& status) const;
    void result(float* out16) const;
};

}  // namespace pclcompat
