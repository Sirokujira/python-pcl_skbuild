// The rest of pcl/registration behind Cython-expressible functions and
// classes; the mirror header states each contract.
//
// Correspondences cross as the flat (pairs, distances) pair of
// registration_args.h (pairs = source, target, source, target, ...;
// distances squared), transforms as column-major float[16], and every
// estimator, rejector or framework is built, configured and run here, so
// none of their template parameters, Eigen arguments, boost graphs or
// shared_ptr-to-const inputs has to reach a pxd.
//
// Not here, and listed in the README: CorrespondenceRejectorFeatures
// (its FeatureContainer calls a getClassName() it does not have, so it
// does not compile for any feature type in 1.14) and
// CorrespondenceRejectorSampleConsensus2D / SACMODEL_REGISTRATION_2D
// (they read .u and .v from points that also need x, y, z, a point type
// PCL does not define).
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_estimation_organized_projection.h>
#include <pcl/registration/correspondence_rejection_organized_boundary.h>
#include <pcl/registration/correspondence_rejection_poly.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/elch.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/incremental_registration.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/lum.h>
#include <pcl/registration/meta_registration.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/pyramid_feature_matching.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/registration/transformation_estimation_symmetric_point_to_plane_lls.h>
#include <pcl/features/ppf.h>

#include "pcl/compat/registration_args.h"

namespace pclcompat {

using PNPoint = pcl::PointNormal;
using PNCloud = pcl::PointCloud<PNPoint>;
using PNCloudPtr = std::shared_ptr<PNCloud>;
using RGBAPoint = pcl::PointXYZRGBA;
using RGBACloud = pcl::PointCloud<RGBAPoint>;
using RGBACloudPtr = std::shared_ptr<RGBACloud>;

template <typename Estimator, typename Points>
inline void runEstimation(Estimator& estimator, const Points& source,
                          const Points& target, const std::vector<int>& pairs,
                          Eigen::Matrix4f& m) {
    const std::vector<float> no_distances;
    if (pairs.empty()) {
        estimator.estimateRigidTransformation(source, target, m);
    } else {
        estimator.estimateRigidTransformation(
            source, target, toCorrespondences(pairs, no_distances), m);
    }
}

// --- transformation estimation ------------------------------------------

inline bool estimateTransformationExtra(int method, const CloudPtr& source,
                                        const CloudPtr& target,
                                        const std::vector<int>& pairs,
                                        float* out_matrix16) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    switch (method) {
    case 2: {
        pcl::registration::TransformationEstimationDualQuaternion<Point, Point> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 3: {
        pcl::registration::TransformationEstimationSVDScale<Point, Point> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 4: {
        pcl::registration::TransformationEstimation2D<Point, Point> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 5: {
        pcl::registration::TransformationEstimation3Point<Point, Point> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    default:
        return false;
    }
    copyMatrix(m, out_matrix16);
    return true;
}

inline bool estimateTransformationNormals(int method, const PNCloudPtr& source,
                                          const PNCloudPtr& target,
                                          const std::vector<int>& pairs,
                                          const std::vector<double>& weights,
                                          float* out_matrix16) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    switch (method) {
    case 0: {
        pcl::registration::TransformationEstimationPointToPlane<PNPoint, PNPoint> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 1: {
        pcl::registration::TransformationEstimationPointToPlaneLLS<PNPoint, PNPoint> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 2: {
        pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<PNPoint, PNPoint> e;
        e.setCorrespondenceWeights(std::vector<float>(weights.begin(), weights.end()));
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 3: {
        pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<PNPoint, PNPoint> e;
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    case 4: {
        pcl::registration::TransformationEstimationPointToPlaneWeighted<PNPoint, PNPoint> e;
        e.setWeights(weights);
        runEstimation(e, *source, *target, pairs, m);
        break;
    }
    default:
        return false;
    }
    copyMatrix(m, out_matrix16);
    return true;
}

// --- correspondence estimation ------------------------------------------

template <typename Estimation>
inline void finishEstimation(Estimation& estimation, double max_distance,
                             std::vector<int>& pairs,
                             std::vector<float>& distances) {
    pcl::Correspondences found;
    estimation.determineCorrespondences(
        found, max_distance > 0 ? max_distance
                                : std::numeric_limits<double>::max());
    fromCorrespondences(found, pairs, distances);
}

inline void correspondencesNormalShooting(const CloudPtr& source,
                                          const NormalCloudPtr& source_normals,
                                          const CloudPtr& target, int k,
                                          double max_distance,
                                          std::vector<int>& pairs,
                                          std::vector<float>& distances) {
    pcl::registration::CorrespondenceEstimationNormalShooting<Point, Point, pcl::Normal> e;
    e.setInputSource(source);
    e.setSourceNormals(source_normals);
    e.setInputTarget(target);
    e.setKSearch(static_cast<unsigned int>(k));
    finishEstimation(e, max_distance, pairs, distances);
}

inline void correspondencesBackProjection(const CloudPtr& source,
                                          const NormalCloudPtr& source_normals,
                                          const CloudPtr& target,
                                          const NormalCloudPtr& target_normals,
                                          int k, double max_distance,
                                          std::vector<int>& pairs,
                                          std::vector<float>& distances) {
    pcl::registration::CorrespondenceEstimationBackProjection<Point, Point, pcl::Normal> e;
    e.setInputSource(source);
    e.setSourceNormals(source_normals);
    e.setInputTarget(target);
    e.setTargetNormals(target_normals);
    e.setKSearch(static_cast<unsigned int>(k));
    finishEstimation(e, max_distance, pairs, distances);
}

inline void correspondencesOrganizedProjection(
    const CloudPtr& source, const CloudPtr& target, float fx, float fy,
    float cx, float cy, float depth_threshold, double max_distance,
    std::vector<int>& pairs, std::vector<float>& distances) {
    pcl::registration::CorrespondenceEstimationOrganizedProjection<Point, Point> e;
    e.setInputSource(source);
    e.setInputTarget(target);
    e.setFocalLengths(fx, fy);
    e.setCameraCenters(cx, cy);
    e.setDepthThreshold(depth_threshold);
    finishEstimation(e, max_distance, pairs, distances);
}

// --- correspondence rejection -------------------------------------------

inline void rejectVarTrimmed(const CloudPtr& source, const CloudPtr& target,
                             const std::vector<int>& pairs,
                             const std::vector<float>& distances,
                             double min_ratio, double max_ratio,
                             std::vector<int>& out_pairs,
                             std::vector<float>& out_distances,
                             double* out_trimmed_distance,
                             double* out_trim_factor) {
    pcl::registration::CorrespondenceRejectorVarTrimmed rejector;
    rejector.setInputSource<Point>(source);
    rejector.setInputTarget<Point>(target);
    rejector.setMinRatio(min_ratio);
    rejector.setMaxRatio(max_ratio);
    pcl::Correspondences kept;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances), kept);
    fromCorrespondences(kept, out_pairs, out_distances);
    *out_trimmed_distance = rejector.getTrimmedDistance();
    *out_trim_factor = rejector.getTrimFactor();
}

inline void rejectPoly(const CloudPtr& source, const CloudPtr& target,
                       const std::vector<int>& pairs,
                       const std::vector<float>& distances, int cardinality,
                       float similarity_threshold, int iterations,
                       std::vector<int>& out_pairs,
                       std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorPoly<Point, Point> rejector;
    rejector.setInputSource(source);
    rejector.setInputTarget(target);
    rejector.setCardinality(cardinality);
    rejector.setSimilarityThreshold(similarity_threshold);
    rejector.setIterations(iterations);
    pcl::Correspondences kept;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances), kept);
    fromCorrespondences(kept, out_pairs, out_distances);
}

inline void rejectOrganizedBoundary(const CloudPtr& source,
                                    const CloudPtr& target,
                                    const std::vector<int>& pairs,
                                    const std::vector<float>& distances,
                                    int boundary_nans,
                                    std::vector<int>& out_pairs,
                                    std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectionOrganizedBoundary rejector;
    rejector.setNumberOfBoundaryNaNs(boundary_nans);
    rejector.setInputSource<Point>(source);
    rejector.setInputTarget<Point>(target);
    pcl::Correspondences kept;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances), kept);
    fromCorrespondences(kept, out_pairs, out_distances);
}

// --- align-style registrations ------------------------------------------

template <typename Registration, typename Cloud>
inline bool finishAlign(Registration& reg, Cloud& out, float* out_matrix16,
                        double* out_fitness) {
    reg.align(out);
    copyMatrix(reg.getFinalTransformation(), out_matrix16);
    *out_fitness = reg.getFitnessScore();
    return reg.hasConverged();
}

inline bool gicp6dAlign(const RGBACloudPtr& source, const RGBACloudPtr& target,
                        float lab_weight, int max_iterations,
                        double max_correspondence_distance,
                        double transformation_epsilon, RGBACloud& out,
                        float* out_matrix16, double* out_fitness) {
    pcl::GeneralizedIterativeClosestPoint6D gicp(lab_weight);
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    gicp.setMaximumIterations(max_iterations);
    gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp.setTransformationEpsilon(transformation_epsilon);
    return finishAlign(gicp, out, out_matrix16, out_fitness);
}

inline bool ndt2dAlign(const CloudPtr& source, const CloudPtr& target,
                       const float* grid_centre2, const float* grid_extent2,
                       const float* grid_step2, const double* optimisation_step3,
                       int max_iterations, double transformation_epsilon,
                       Cloud& out, float* out_matrix16, double* out_fitness) {
    pcl::NormalDistributionsTransform2D<Point, Point> ndt;
    ndt.setInputSource(source);
    ndt.setInputTarget(target);
    ndt.setGridCentre(Eigen::Vector2f(grid_centre2[0], grid_centre2[1]));
    ndt.setGridExtent(Eigen::Vector2f(grid_extent2[0], grid_extent2[1]));
    ndt.setGridStep(Eigen::Vector2f(grid_step2[0], grid_step2[1]));
    ndt.setOptimizationStepSize(Eigen::Vector3d(
        optimisation_step3[0], optimisation_step3[1], optimisation_step3[2]));
    ndt.setMaximumIterations(max_iterations);
    ndt.setTransformationEpsilon(transformation_epsilon);
    return finishAlign(ndt, out, out_matrix16, out_fitness);
}

inline bool jointICPAlign(const std::vector<CloudPtr>& sources,
                          const std::vector<CloudPtr>& targets,
                          int max_iterations,
                          double max_correspondence_distance,
                          double transformation_epsilon, float* out_matrix16,
                          double* out_fitness) {
    pcl::JointIterativeClosestPoint<Point, Point> icp;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        icp.addInputSource(sources[i]);
        icp.addInputTarget(targets[i]);
    }
    icp.setMaximumIterations(max_iterations);
    icp.setMaxCorrespondenceDistance(max_correspondence_distance);
    icp.setTransformationEpsilon(transformation_epsilon);
    Cloud out;
    icp.align(out);
    copyMatrix(icp.getFinalTransformation(), out_matrix16);
    *out_fitness = icp.getFitnessScore();
    return icp.hasConverged();
}

inline bool ppfRegister(const PNCloudPtr& model, const PNCloudPtr& scene,
                        float angle_step, float distance_step,
                        unsigned int scene_sampling_rate,
                        float position_threshold, float rotation_threshold,
                        PNCloud& out, float* out_matrix16,
                        double* out_fitness) {
    pcl::PointCloud<pcl::PPFSignature>::Ptr model_features(
        new pcl::PointCloud<pcl::PPFSignature>());
    pcl::PPFEstimation<PNPoint, PNPoint, pcl::PPFSignature> estimation;
    estimation.setInputCloud(model);
    estimation.setInputNormals(model);
    estimation.compute(*model_features);

    pcl::PPFHashMapSearch::Ptr search(
        new pcl::PPFHashMapSearch(angle_step, distance_step));
    search->setInputFeatureCloud(model_features);

    pcl::PPFRegistration<PNPoint, PNPoint> ppf;
    ppf.setSceneReferencePointSamplingRate(scene_sampling_rate);
    ppf.setPositionClusteringThreshold(position_threshold);
    ppf.setRotationClusteringThreshold(rotation_threshold);
    ppf.setSearchMethod(search);
    ppf.setInputSource(model);
    ppf.setInputTarget(scene);
    return finishAlign(ppf, out, out_matrix16, out_fitness);
}

// PyramidFeatureHistogram is dense -- every level allocates the product
// of its per-dimension bin counts -- so only low-dimensional features are
// usable (2 bins over FPFH's 33 dimensions is already 2^33 counters).
// The features ride in an FPFHSignature33 (n x dims, flat, dims <= 16)
// and this representation hands the pyramid the leading `dims` values.
// at() does not bounds-check, so the caller keeps every value inside the
// input range and the bin total small; the Python side enforces both.
class LeadingDimensions
    : public pcl::PointRepresentation<pcl::FPFHSignature33> {
public:
    explicit LeadingDimensions(int dims) {
        nr_dimensions_ = dims;
        trivial_ = false;
    }

    void copyToFloatArray(const pcl::FPFHSignature33& p,
                          float* out) const override {
        for (int i = 0; i < nr_dimensions_; ++i) {
            out[i] = p.histogram[i];
        }
    }
};

inline std::shared_ptr<pcl::PointCloud<pcl::FPFHSignature33>> featureCloud(
    const std::vector<float>& flat, int dims) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
    cloud->resize(flat.size() / dims);
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        for (int j = 0; j < dims; ++j) {
            (*cloud)[i].histogram[j] = flat[dims * i + j];
        }
    }
    return cloud;
}

// a, b: n x dims and m x dims, flat; input_ranges / target_ranges: dims
// (min, max) pairs, flat. Similarity in [0, 1].
inline float pyramidSimilarity(int dims, const std::vector<float>& a,
                               const std::vector<float>& b,
                               const std::vector<float>& input_ranges,
                               const std::vector<float>& target_ranges) {
    std::vector<std::pair<float, float>> input, target;
    for (int i = 0; i < dims; ++i) {
        input.emplace_back(input_ranges[2 * i], input_ranges[2 * i + 1]);
        target.emplace_back(target_ranges[2 * i], target_ranges[2 * i + 1]);
    }
    using Pyramid = pcl::PyramidFeatureHistogram<pcl::FPFHSignature33>;
    auto representation = std::make_shared<const LeadingDimensions>(dims);
    auto build = [&](const std::vector<float>& flat) {
        typename Pyramid::Ptr pyramid(new Pyramid());
        pyramid->setPointRepresentation(representation);
        pyramid->setInputDimensionRange(input);
        pyramid->setTargetDimensionRange(target);
        pyramid->setInputCloud(featureCloud(flat, dims));
        pyramid->compute();
        return pyramid;
    };
    return Pyramid::comparePyramidFeatureHistograms(build(a), build(b));
}

// --- stateful frameworks --------------------------------------------------

inline std::shared_ptr<pcl::IterativeClosestPoint<Point, Point>> makeICP(
    double max_correspondence_distance, int max_iterations,
    double transformation_epsilon) {
    auto icp = std::make_shared<pcl::IterativeClosestPoint<Point, Point>>();
    icp->setMaxCorrespondenceDistance(max_correspondence_distance);
    icp->setMaximumIterations(max_iterations);
    icp->setTransformationEpsilon(transformation_epsilon);
    return icp;
}

class IncrementalICP {
public:
    IncrementalICP(double max_correspondence_distance, int max_iterations,
                   double transformation_epsilon) {
        reg_.setRegistration(makeICP(max_correspondence_distance,
                                     max_iterations, transformation_epsilon));
    }
    bool registerCloud(const CloudPtr& cloud, const float* delta16) {
        const Eigen::Matrix4f delta =
            delta16 ? readMatrix(delta16) : Eigen::Matrix4f::Identity();
        return reg_.registerCloud(cloud, delta);
    }
    void deltaTransform(float* out16) const { copyMatrix(reg_.getDeltaTransform(), out16); }
    void absoluteTransform(float* out16) const { copyMatrix(reg_.getAbsoluteTransform(), out16); }
    void reset() { reg_.reset(); }

private:
    pcl::registration::IncrementalRegistration<Point> reg_;
};

class MetaICP {
public:
    MetaICP(double max_correspondence_distance, int max_iterations,
            double transformation_epsilon) {
        reg_.setRegistration(makeICP(max_correspondence_distance,
                                     max_iterations, transformation_epsilon));
    }
    bool registerCloud(const CloudPtr& cloud, const float* delta16) {
        const Eigen::Matrix4f delta =
            delta16 ? readMatrix(delta16) : Eigen::Matrix4f::Identity();
        return reg_.registerCloud(cloud, delta);
    }
    void absoluteTransform(float* out16) const { copyMatrix(reg_.getAbsoluteTransform(), out16); }
    void metaCloud(Cloud& out) const { out = *reg_.getMetaCloud(); }
    void reset() { reg_.reset(); }

private:
    pcl::registration::MetaRegistration<Point> reg_;
};

class LUMGraph {
public:
    int addCloud(const CloudPtr& cloud, const float* pose6) {
        Eigen::Vector6f pose = Eigen::Vector6f::Zero();
        if (pose6) {
            for (int i = 0; i < 6; ++i) {
                pose[i] = pose6[i];
            }
        }
        return static_cast<int>(lum_.addPointCloud(cloud, pose));
    }
    void setCorrespondences(int source, int target, const std::vector<int>& pairs) {
        auto c = std::make_shared<pcl::Correspondences>(
            toCorrespondences(pairs, std::vector<float>()));
        lum_.setCorrespondences(source, target, c);
    }
    void setMaxIterations(int count) { lum_.setMaxIterations(count); }
    void setConvergenceThreshold(float threshold) { lum_.setConvergenceThreshold(threshold); }
    void compute() { lum_.compute(); }
    int size() const { return static_cast<int>(lum_.getNumVertices()); }
    void pose(int vertex, float* out6) const {
        const Eigen::Vector6f p = lum_.getPose(vertex);
        for (int i = 0; i < 6; ++i) {
            out6[i] = p[i];
        }
    }
    void transformation(int vertex, float* out16) const {
        copyMatrix(lum_.getTransformation(vertex).matrix(), out16);
    }
    void transformedCloud(int vertex, Cloud& out) const { out = *lum_.getTransformedCloud(vertex); }
    void concatenatedCloud(Cloud& out) const { out = *lum_.getConcatenatedCloud(); }

private:
    pcl::registration::LUM<Point> lum_;
};

class ELCHLoop {
public:
    ELCHLoop(double max_correspondence_distance, int max_iterations,
             double transformation_epsilon) {
        elch_.setReg(makeICP(max_correspondence_distance, max_iterations,
                             transformation_epsilon));
    }
    int addCloud(const CloudPtr& cloud) {
        // ELCH transforms the clouds in its graph in place; hand it a copy.
        elch_.addPointCloud(std::make_shared<Cloud>(*cloud));
        return count_++;
    }
    void setLoopStart(int vertex) { elch_.setLoopStart(vertex); }
    void setLoopEnd(int vertex) { elch_.setLoopEnd(vertex); }
    void setLoopTransform(const float* matrix16) { elch_.setLoopTransform(readMatrix(matrix16)); }
    void compute() { elch_.compute(); }
    int size() const { return count_; }
    void cloud(int vertex, Cloud& out) { out = *(*elch_.getLoopGraph())[vertex].cloud; }

private:
    pcl::registration::ELCH<Point> elch_;
    int count_ = 0;
};

}  // namespace pclcompat
