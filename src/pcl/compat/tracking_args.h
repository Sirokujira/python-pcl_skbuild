// Particle-filter tracking behind one Cython-expressible class.
//
// PCL's tracker needs five objects wired together before it does
// anything: the tracker itself, a PointCloudCoherence, a PointCoherence
// added to it, a search method handed to that, and an Eigen::Affine3f
// initial pose. Every one of those travels as a shared_ptr to a class
// template, and the tracker's result is a `ParticleXYZRPY` — a point
// type whose fields live in an anonymous union. `ParticleTracker` owns
// the lot and exposes floats.
//
// It tracks the OMP variant on purpose. `ParticleFilterTracker` declares
// `bool changed_{false}` and NEVER assigns it; only the OMP subclasses
// do. Since `computeTracking()` gates both `resample()` and `update()`
// on that flag, the plain tracker computes weights and then throws them
// away — getResult() returns the initial state forever. Verified in
// plain C++ against PCL 1.14: the base tracker reports (0,0,0) on every
// frame of a moving object, the OMP one follows it.
#pragma once

#include <memory>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/octree.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/tracking.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using ParticleState = pcl::tracking::ParticleXYZRPY;

class ParticleTracker {
public:
    ParticleTracker(int particle_num, double step_noise, double resolution,
                    double maximum_distance, int threads)
        : tracker_(new pcl::tracking::ParticleFilterOMPTracker<
                       Point, ParticleState>(threads)) {
        configure(particle_num, step_noise, resolution, maximum_distance);
    }

    /// The KLD-adaptive variant: the particle count adapts between
    /// `particle_num` and `max_particle_num`, growing while the particles
    /// spread over more state-space bins of size `bin_size` (one value
    /// for all six dimensions) than the KLD bound for `epsilon` / `delta`
    /// allows. Also the OMP class, for the same `changed_` reason.
    ParticleTracker(int particle_num, double step_noise, double resolution,
                    double maximum_distance, int threads,
                    int max_particle_num, float bin_size, double epsilon,
                    double delta) {
        auto kld = std::make_shared<
            pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<
                Point, ParticleState>>(threads);
        kld->setMaximumParticleNum(max_particle_num);
        kld->setEpsilon(epsilon);
        kld->setDelta(delta);
        ParticleState bins;
        bins.x = bins.y = bins.z = bin_size;
        bins.roll = bins.pitch = bins.yaw = bin_size;
        kld->setBinSize(bins);
        tracker_ = kld;
        configure(particle_num, step_noise, resolution, maximum_distance);
    }

private:
    void configure(int particle_num, double step_noise, double resolution,
                   double maximum_distance) {
        tracker_->setParticleNum(particle_num);
        tracker_->setIterationNum(1);
        tracker_->setResampleLikelihoodThr(0.0);
        // The change detector skips the whole update when the scene
        // looks unchanged, which turns a stationary-camera test into a
        // silent no-op. Callers who want it can set it back on.
        tracker_->setUseChangeDetector(false);
        tracker_->setUseNormal(false);

        const std::vector<double> step(6, step_noise);
        tracker_->setStepNoiseCovariance(step);
        tracker_->setInitialNoiseCovariance(std::vector<double>(6, 0.00001));
        tracker_->setInitialNoiseMean(std::vector<double>(6, 0.0));

        auto coherence = std::make_shared<
            pcl::tracking::ApproxNearestPairPointCloudCoherence<Point>>();
        coherence->addPointCoherence(
            std::make_shared<pcl::tracking::DistanceCoherence<Point>>());
        coherence->setSearchMethod(
            std::make_shared<pcl::search::Octree<Point>>(resolution));
        coherence->setMaximumDistance(maximum_distance);
        tracker_->setCloudCoherence(coherence);
    }

public:
    /// The reference cloud is the object to follow. PCL wants it in the
    /// object's own frame with the pose supplied separately, so the
    /// centroid is subtracted here and handed back as the initial
    /// transform — otherwise a particle's rotation swings the model
    /// around the world origin instead of its own centre.
    void setReferenceCloud(const std::shared_ptr<Cloud>& reference) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid<Point>(*reference, centroid);

        Eigen::Affine3f trans = Eigen::Affine3f::Identity();
        trans.translation() << centroid[0], centroid[1], centroid[2];

        std::shared_ptr<Cloud> centered(new Cloud());
        pcl::transformPointCloud<Point>(*reference, *centered, trans.inverse());

        tracker_->setReferenceCloud(centered);
        tracker_->setTrans(trans);
    }

    void setInputCloud(const std::shared_ptr<Cloud>& cloud) {
        tracker_->setInputCloud(cloud);
    }

    void compute() { tracker_->compute(); }

    /// x, y, z, roll, pitch, yaw, weight — the whole particle state.
    void result(float* out7) const {
        const ParticleState state = tracker_->getResult();
        out7[0] = state.x;
        out7[1] = state.y;
        out7[2] = state.z;
        out7[3] = state.roll;
        out7[4] = state.pitch;
        out7[5] = state.yaw;
        out7[6] = state.weight;
    }

    /// The same pose as a 4x4 matrix, column-major (Eigen's own storage,
    /// which is the Fortran order numpy wants).
    void resultTransform(float* out16) const {
        const Eigen::Affine3f trans =
            tracker_->toEigenMatrix(tracker_->getResult());
        const float* data = trans.matrix().data();
        for (int i = 0; i < 16; ++i) {
            out16[i] = data[i];
        }
    }

    /// The reference cloud placed at the tracked pose — what a caller
    /// draws over the scene to see the fit.
    void alignedReference(std::vector<Point>& out) const {
        Cloud aligned;
        pcl::transformPointCloud<Point>(
            *tracker_->getReferenceCloud(), aligned,
            tracker_->toEigenMatrix(tracker_->getResult()));
        out.assign(aligned.points.begin(), aligned.points.end());
    }

    int particleNum() const { return tracker_->getParticleNum(); }

private:
    // The base class: both variants are driven through the same calls,
    // and everything they override (initCompute, resample,
    // computeTracking, weight) is virtual.
    std::shared_ptr<
        pcl::tracking::ParticleFilterTracker<Point, ParticleState>> tracker_;
};

}  // namespace pclcompat
