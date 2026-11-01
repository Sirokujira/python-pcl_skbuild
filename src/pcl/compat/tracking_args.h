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
//
// The tracker is a template over the point type so colour (HSV
// coherence on PointXYZRGBA) and normals (normal coherence on
// PointNormal) can join the distance coherence; `ParticleTracker` is the
// PointXYZ instance the original wrapper used. PyramidalKLTTracker
// (2-D keypoints followed across organized colour frames) is at the end,
// with three of PCL's mistakes corrected -- see CorrectedKLT.
#pragma once

#include <memory>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/octree.h>
#include <pcl/pcl_config.h>
#include <pcl/search/kdtree.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/normal_coherence.h>
// pyramidal_klt.h names pcl::InterpolationType without including
// pcl/common/io.h, which declares it (still so on master).
#include <pcl/common/io.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/tracking/pyramidal_klt.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/tracking.h>

namespace pclcompat {

using ParticleState = pcl::tracking::ParticleXYZRPY;

/// The reference placed at a pose; PointNormal rotates its normals too.
template <typename PointT>
inline void placeCloud(const pcl::PointCloud<PointT>& in,
                       pcl::PointCloud<PointT>& out,
                       const Eigen::Affine3f& pose) {
    pcl::transformPointCloud<PointT>(in, out, pose);
}

inline void placeCloud(const pcl::PointCloud<pcl::PointNormal>& in,
                       pcl::PointCloud<pcl::PointNormal>& out,
                       const Eigen::Affine3f& pose) {
    pcl::transformPointCloudWithNormals<pcl::PointNormal>(in, out, pose);
}

/// The OMP trackers set `changed_` -- the flag computeTracking() gates
/// resample() and update() on -- only on their no-normals path, so with
/// setUseNormal(true) they weigh the particles and drop the weights,
/// the base class's fault again (1.11 through master; plain C++: a box
/// moved 8 cm reads 0.0000 stock, 0.0843 with this).
template <typename Tracker>
class WeighingNormals : public Tracker {
public:
    using Tracker::Tracker;

protected:
    void weight() override {
        Tracker::weight();
        if (this->use_normal_) {
            this->changed_ = true;
        }
    }
};

template <typename PointT>
class ParticleTrackerT {
    using Point = PointT;
    using Cloud = pcl::PointCloud<Point>;

public:
    ParticleTrackerT(int particle_num, double step_noise, double resolution,
                     double maximum_distance, int threads)
        : tracker_(new WeighingNormals<pcl::tracking::ParticleFilterOMPTracker<
                       Point, ParticleState>>(threads)) {
        configure(particle_num, step_noise, resolution, maximum_distance);
    }

    /// The KLD-adaptive variant: the particle count adapts between
    /// `particle_num` and `max_particle_num`, growing while the particles
    /// spread over more state-space bins of size `bin_size` (one value
    /// for all six dimensions) than the KLD bound for `epsilon` / `delta`
    /// allows. Also the OMP class, for the same `changed_` reason.
    ParticleTrackerT(int particle_num, double step_noise, double resolution,
                     double maximum_distance, int threads,
                     int max_particle_num, float bin_size, double epsilon,
                     double delta) {
        auto kld = std::make_shared<WeighingNormals<
            pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<
                Point, ParticleState>>>(threads);
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

        resolution_ = resolution;
        maximum_distance_ = maximum_distance;
        point_coherences_.assign(
            1, std::make_shared<pcl::tracking::DistanceCoherence<Point>>());
        buildCloudCoherence();
    }

    /// Approximate (octree voxel) or exact (kd-tree) nearest pairs, with
    /// every point coherence added so far.
    void buildCloudCoherence() {
        std::shared_ptr<pcl::tracking::NearestPairPointCloudCoherence<Point>>
            coherence;
        if (exact_) {
            coherence = std::make_shared<
                pcl::tracking::NearestPairPointCloudCoherence<Point>>();
            coherence->setSearchMethod(
                std::make_shared<pcl::search::KdTree<Point>>(false));
        } else {
            auto approx = std::make_shared<
                pcl::tracking::ApproxNearestPairPointCloudCoherence<Point>>();
            approx->setSearchMethod(
                std::make_shared<pcl::search::Octree<Point>>(resolution_));
            coherence = approx;
        }
        for (const auto& point : point_coherences_) {
            coherence->addPointCoherence(point);
        }
        coherence->setMaximumDistance(maximum_distance_);
        tracker_->setCloudCoherence(coherence);
    }

public:
    /// Exact nearest pairs (NearestPairPointCloudCoherence over a kd-tree)
    /// instead of the octree-voxel approximation; slower, finer.
    void setExactCoherence(bool exact) {
        exact_ = exact;
        buildCloudCoherence();
    }

    /// Add a point coherence (colour, normals) next to the distance one.
    void addPointCoherence(
        const std::shared_ptr<pcl::tracking::PointCoherence<Point>>& point) {
        point_coherences_.push_back(point);
        buildCloudCoherence();
    }

    void setUseNormal(bool use) { tracker_->setUseNormal(use); }

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
        alignedReferenceCloud(aligned);
        out.assign(aligned.points.begin(), aligned.points.end());
    }

    void alignedReferenceCloud(Cloud& out) const {
        placeCloud(*tracker_->getReferenceCloud(), out,
                   tracker_->toEigenMatrix(tracker_->getResult()));
    }

    int particleNum() const { return tracker_->getParticleNum(); }

private:
    // The base class: both variants are driven through the same calls,
    // and everything they override (initCompute, resample,
    // computeTracking, weight) is virtual.
    std::shared_ptr<
        pcl::tracking::ParticleFilterTracker<Point, ParticleState>> tracker_;
    std::vector<std::shared_ptr<pcl::tracking::PointCoherence<Point>>>
        point_coherences_;
    double resolution_ = 0.01;
    double maximum_distance_ = 0.05;
    bool exact_ = false;
};

using ParticleTracker = ParticleTrackerT<pcl::PointXYZ>;
using ParticleTrackerRGBA = ParticleTrackerT<pcl::PointXYZRGBA>;
using ParticleTrackerNormal = ParticleTrackerT<pcl::PointNormal>;

/// HSV colour coherence (weights for hue, saturation, value and the
/// whole term; PCL's defaults are 1, 1, 0 and 1).
inline void addHSVColorCoherence(ParticleTrackerRGBA& tracker, double h,
                                 double s, double v, double weight) {
    auto coherence = std::make_shared<
        pcl::tracking::HSVColorCoherence<pcl::PointXYZRGBA>>();
    coherence->setHWeight(h);
    coherence->setSWeight(s);
    coherence->setVWeight(v);
    coherence->setWeight(weight);
    tracker.addPointCoherence(coherence);
}

/// Normal coherence; the tracker then moves the reference's normals with
/// each particle (setUseNormal).
inline void addNormalCoherence(ParticleTrackerNormal& tracker, double weight) {
    auto coherence = std::make_shared<
        pcl::tracking::NormalCoherence<pcl::PointNormal>>();
    coherence->setWeight(weight);
    tracker.addPointCoherence(coherence);
    tracker.setUseNormal(true);
}

// --- PyramidalKLTTracker -----------------------------------------------------

/// pcl::tracking::PyramidalKLTTracker with its two virtual stages
/// replaced. Stock PCL (1.11 through master) barely moves a point: a
/// 2-pixel shift reads as 0.5-0.9 and a 12-pixel one as about 1, at any
/// pyramid depth (plain C++ on synthetic textures). Three separate faults:
///
/// - computePyramids() pads level 0's image, takes its gradients, and
///   then pads those AGAIN, so they are 2 * track_width wider than the
///   image that track() indexes them with -- read with the wrong stride.
/// - The gradients are raw Scharr output, 32x a unit derivative, while
///   the intensity differences are not scaled to match (OpenCV, which
///   this ports, shifts the image by 5 bits for exactly that), so every
///   Lucas-Kanade step is 1/32 of what it should be and the epsilon test
///   stops it after the first.
/// - track() places windows in the padded images at the unpadded
///   coordinate, i.e. track_width * 2^level pixels up and left of the
///   requested point (read past the right edge at the other end).
///
/// The overrides take each level's gradients from its padded image,
/// divided by 32, and add the padding back when placing windows; the rest
/// of track() is PCL's, plus a NaN guard on the 3-D correspondences. With
/// them the same synthetic shifts and a 3-degree rotation track to within
/// 0.02 px in the interior; points whose content leaves the frame can
/// still lock onto the wrong place, as in any KLT without an error check.
class CorrectedKLT
    : public pcl::tracking::PyramidalKLTTracker<pcl::PointXYZRGBA> {
    using Base = pcl::tracking::PyramidalKLTTracker<pcl::PointXYZRGBA>;

public:
    using Base::Base;

protected:
    void computePyramids(const PointCloudInConstPtr& input,
                         std::vector<FloatImageConstPtr>& pyramid,
                         pcl::InterpolationType border) const override {
        Base::computePyramids(input, pyramid, border);
        for (int level = 0; level < nb_levels_; ++level) {
            FloatImagePtr gx(new FloatImage), gy(new FloatImage);
            derivatives(*pyramid[level * 3], *gx, *gy);
            for (auto& g : *gx) g /= 32.f;
            for (auto& g : *gy) g /= 32.f;
            pyramid[level * 3 + 1] = gx;
            pyramid[level * 3 + 2] = gy;
        }
    }

    void track(const PointCloudInConstPtr& prev_input,
               const PointCloudInConstPtr& input,
               const std::vector<FloatImageConstPtr>& prev_pyramid,
               const std::vector<FloatImageConstPtr>& pyramid,
               const pcl::PointCloud<pcl::PointUV>::ConstPtr& prev_keypoints,
               pcl::PointCloud<pcl::PointUV>::Ptr& keypoints,
               std::vector<int>& status,
               Eigen::Affine3f& motion) const override {
        const int n = static_cast<int>(prev_keypoints->size());
        std::vector<Eigen::Array2f, Eigen::aligned_allocator<Eigen::Array2f>>
            next_pts(n);
        const Eigen::Array2f half_win((track_width_ - 1) * 0.5f,
                                      (track_height_ - 1) * 0.5f);
        const Eigen::Array2f pad(static_cast<float>(track_width_),
                                 static_cast<float>(track_height_));
        // a window's top-left corner, in padded pixels, fits when the
        // window and its +1 interpolation neighbour do
        const auto inside = [this](const Eigen::Array2i& corner,
                                   const FloatImage& image) {
            return corner[0] >= 0 && corner[1] >= 0 &&
                   corner[0] + track_width_ + 1 <
                       static_cast<int>(image.width) &&
                   corner[1] + track_height_ + 1 <
                       static_cast<int>(image.height);
        };
        const auto weights = [](float a, float b) {
            Eigen::Array4f w((1.f - a) * (1.f - b), a * (1.f - b),
                             (1.f - a) * b, 0.f);
            w[3] = 1 - w[0] - w[1] - w[2];
            return w;
        };
        for (int level = nb_levels_ - 1; level >= 0; --level) {
            const FloatImage& prev = *prev_pyramid[level * 3];
            const FloatImage& next = *pyramid[level * 3];
            const FloatImage& grad_x = *prev_pyramid[level * 3 + 1];
            const FloatImage& grad_y = *prev_pyramid[level * 3 + 2];
            Eigen::ArrayXXf prev_win(track_height_, track_width_);
            Eigen::ArrayXXf grad_x_win(track_height_, track_width_);
            Eigen::ArrayXXf grad_y_win(track_height_, track_width_);
            const float ratio = 1.f / (1 << level);
            for (int i = 0; i < n; ++i) {
                if (status[i] != 0) {
                    continue;
                }
                Eigen::Array2f prev_pt((*prev_keypoints)[i].u * ratio,
                                       (*prev_keypoints)[i].v * ratio);
                Eigen::Array2f next_pt = level == nb_levels_ - 1
                                             ? prev_pt
                                             : Eigen::Array2f(next_pts[i] * 2.f);
                next_pts[i] = next_pt;

                prev_pt = prev_pt - half_win + pad;
                const Eigen::Array2i corner(
                    static_cast<int>(std::floor(prev_pt[0])),
                    static_cast<int>(std::floor(prev_pt[1])));
                if (!inside(corner, prev)) {
                    if (level == 0) {
                        status[i] = -1;
                    }
                    continue;
                }
                Eigen::Array4f weight = weights(prev_pt[0] - corner[0],
                                                prev_pt[1] - corner[1]);
                Eigen::Array3f covar;
                spatialGradient(prev, grad_x, grad_y, corner, weight, prev_win,
                                grad_x_win, grad_y_win, covar);
                float det = covar[0] * covar[2] - covar[1] * covar[1];
                const float min_eigenvalue =
                    (covar[2] + covar[0] -
                     std::sqrt((covar[0] - covar[2]) * (covar[0] - covar[2]) +
                               4.f * covar[1] * covar[1])) /
                    2.f;
                if (min_eigenvalue < min_eigenvalue_threshold_ ||
                    det < std::numeric_limits<float>::epsilon()) {
                    if (level == 0) {
                        status[i] = -2;
                    }
                    continue;
                }
                det = 1.f / det;
                next_pt = next_pt - half_win + pad;
                Eigen::Array2f prev_delta(0, 0);
                for (unsigned int j = 0; j < max_iterations_; ++j) {
                    const Eigen::Array2i at = next_pt.floor().cast<int>();
                    if (!inside(at, next)) {
                        if (level == 0) {
                            status[i] = -1;
                        }
                        break;
                    }
                    weight = weights(next_pt[0] - at[0], next_pt[1] - at[1]);
                    Eigen::Array2f beta = Eigen::Array2f::Zero();
                    mismatchVector(prev_win, grad_x_win, grad_y_win, next, at,
                                   weight, beta);
                    const Eigen::Vector2f delta(
                        (covar[1] * beta[1] - covar[2] * beta[0]) * det,
                        (covar[1] * beta[0] - covar[0] * beta[1]) * det);
                    next_pt[0] += delta[0];
                    next_pt[1] += delta[1];
                    next_pts[i] = next_pt + half_win - pad;
                    if (delta.squaredNorm() <= epsilon_) {
                        break;
                    }
                    if (j > 0 && std::abs(delta[0] + prev_delta[0]) < 0.01 &&
                        std::abs(delta[1] + prev_delta[1]) < 0.01) {
                        next_pts[i][0] -= delta[0] * 0.5f;
                        next_pts[i][1] -= delta[1] * 0.5f;
                        break;
                    }
                    prev_delta = delta;
                }
            }
        }
        // the survivors, in order, and the rigid motion their 3-D points
        // imply
        pcl::TransformationFromCorrespondences correspondences;
        for (int i = 0; i < n; ++i) {
            if (status[i] != 0) {
                continue;
            }
            const Eigen::Array2i to = next_pts[i].floor().cast<int>();
            if (to[0] < 0 || to[1] < 0 ||
                to[0] >= static_cast<int>(input->width) ||
                to[1] >= static_cast<int>(input->height)) {
                status[i] = -1;
                continue;
            }
            pcl::PointUV uv;
            uv.u = next_pts[i][0];
            uv.v = next_pts[i][1];
            keypoints->push_back(uv);
            const int fu = static_cast<int>(std::floor((*prev_keypoints)[i].u));
            const int fv = static_cast<int>(std::floor((*prev_keypoints)[i].v));
            if (fu < 0 || fv < 0 || fu >= static_cast<int>(prev_input->width) ||
                fv >= static_cast<int>(prev_input->height)) {
                continue;
            }
            const auto& p = (*prev_input)(fu, fv);
            const auto& q = (*input)(to[0], to[1]);
            if (pcl::isFinite(p) && pcl::isFinite(q)) {
                correspondences.add(p.getVector3fMap(), q.getVector3fMap(),
                                    1.0);
            }
        }
        motion = correspondences.getTransformation();
    }
};

/// CorrectedKLT on organized colour frames: the first input is the
/// reference, `setPointsToTrack` picks pixels in it, and each later input
/// moves them (pyramidal Lucas-Kanade on the intensity) and estimates the
/// frame-to-frame rigid motion.
class KLTTracker {
public:
    KLTTracker(int levels, int window_width, int window_height)
        : tracker_(levels, window_width, window_height) {}

    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud) {
        tracker_.setInputCloud(cloud);
    }

    /// Pixels to follow, as (u, v) pairs, flat.
    void setPointsToTrack(const std::vector<float>& uv) {
        auto points = std::make_shared<pcl::PointCloud<pcl::PointUV>>();
        for (std::size_t i = 0; i + 1 < uv.size(); i += 2) {
            pcl::PointUV p;
            p.u = uv[i];
            p.v = uv[i + 1];
            points->push_back(p);
        }
        tracker_.setPointsToTrack(points);
    }

    void setNumberOfKeypoints(std::size_t n) { tracker_.setNumberOfKeypoints(n); }
    void setAccuracy(float accuracy) { tracker_.setAccuracy(accuracy); }
    void setEpsilon(float epsilon) { tracker_.setEpsilon(epsilon); }
    void setMaxIterationsNumber(unsigned int n) {
        tracker_.setMaxIterationsNumber(n);
    }
    void setNumberOfThreads(unsigned int n) { tracker_.setNumberOfThreads(n); }

    void compute() { tracker_.compute(); }

    bool initialized() const { return tracker_.getInitialized(); }

    /// Current (u, v) of every tracked point, flat, and its status
    /// (0 tracked; PCL's negative codes otherwise).
    void trackedPoints(std::vector<float>& uv, std::vector<int>& status) const {
        uv.clear();
        status.clear();
        const auto points = tracker_.getTrackedPoints();
        if (points) {
            for (const auto& p : *points) {
                uv.push_back(p.u);
                uv.push_back(p.v);
            }
        }
#if PCL_VERSION_COMPARE(>=, 1, 12, 0)
        const auto codes = tracker_.getStatusOfPointsToTrack();
        if (codes) {
            status.assign(codes->begin(), codes->end());
        }
#else
        const auto codes = tracker_.getPointsToTrackStatus();
        if (codes) {
            status.assign(codes->indices.begin(), codes->indices.end());
        }
#endif
    }

    /// The estimated motion since the reference, column-major 4x4.
    void result(float* out16) const {
        const Eigen::Affine3f motion = tracker_.getResult();
        const float* data = motion.matrix().data();
        for (int i = 0; i < 16; ++i) {
            out16[i] = data[i];
        }
    }

private:
    CorrectedKLT tracker_;
};

}  // namespace pclcompat
