// C++ shim: PCL's correspondence pipeline as plain functions over flat
// index pairs and distances. See pxdgen/headers/pcl/compat/registration_args.h
// for the mirror.
#pragma once

#include <limits>
#include <memory>
#include <vector>

#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_validation_euclidean.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using CloudPtr = std::shared_ptr<Cloud>;
using NormalCloudPtr = std::shared_ptr<pcl::PointCloud<pcl::Normal>>;

inline pcl::Correspondences toCorrespondences(
    const std::vector<int>& pairs, const std::vector<float>& distances) {
    pcl::Correspondences result;
    const std::size_t n = pairs.size() / 2;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float d = i < distances.size()
                            ? distances[i]
                            : std::numeric_limits<float>::max();
        result.emplace_back(pairs[2 * i], pairs[2 * i + 1], d);
    }
    return result;
}

inline void fromCorrespondences(const pcl::Correspondences& correspondences,
                                std::vector<int>& pairs,
                                std::vector<float>& distances) {
    pairs.clear();
    distances.clear();
    pairs.reserve(2 * correspondences.size());
    distances.reserve(correspondences.size());
    for (const auto& c : correspondences) {
        pairs.push_back(static_cast<int>(c.index_query));
        pairs.push_back(static_cast<int>(c.index_match));
        distances.push_back(c.distance);
    }
}

inline void copyMatrix(const Eigen::Matrix4f& m, float* out16) {
    const float* data = m.data();
    for (int i = 0; i < 16; ++i) {
        out16[i] = data[i];
    }
}

inline Eigen::Matrix4f readMatrix(const float* column_major16) {
    Eigen::Matrix4f m;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            m(row, col) = column_major16[col * 4 + row];
        }
    }
    return m;
}

inline void estimateCorrespondences(const CloudPtr& source,
                                    const CloudPtr& target,
                                    double max_distance, bool reciprocal,
                                    std::vector<int>& pairs,
                                    std::vector<float>& distances) {
    pcl::registration::CorrespondenceEstimation<Point, Point> estimation;
    estimation.setInputSource(source);
    estimation.setInputTarget(target);
    const double limit = max_distance > 0
                             ? max_distance
                             : std::numeric_limits<double>::max();
    pcl::Correspondences correspondences;
    if (reciprocal) {
        estimation.determineReciprocalCorrespondences(correspondences, limit);
    } else {
        estimation.determineCorrespondences(correspondences, limit);
    }
    fromCorrespondences(correspondences, pairs, distances);
}

inline void rejectOneToOne(const std::vector<int>& pairs,
                           const std::vector<float>& distances,
                           std::vector<int>& out_pairs,
                           std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorOneToOne rejector;
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
}

inline void rejectDistance(const std::vector<int>& pairs,
                           const std::vector<float>& distances,
                           float max_distance, std::vector<int>& out_pairs,
                           std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorDistance rejector;
    rejector.setMaximumDistance(max_distance);
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
}

inline void rejectMedianDistance(const std::vector<int>& pairs,
                                 const std::vector<float>& distances,
                                 double factor, std::vector<int>& out_pairs,
                                 std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorMedianDistance rejector;
    rejector.setMedianFactor(factor);
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
}

inline void rejectTrimmed(const std::vector<int>& pairs,
                          const std::vector<float>& distances,
                          float overlap_ratio, unsigned int min_correspondences,
                          std::vector<int>& out_pairs,
                          std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorTrimmed rejector;
    rejector.setOverlapRatio(overlap_ratio);
    rejector.setMinCorrespondences(min_correspondences);
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
}

inline bool rejectSampleConsensus(const CloudPtr& source,
                                  const CloudPtr& target,
                                  const std::vector<int>& pairs,
                                  const std::vector<float>& distances,
                                  double inlier_threshold, int max_iterations,
                                  bool refine, std::vector<int>& out_pairs,
                                  std::vector<float>& out_distances,
                                  float* out_matrix16) {
    pcl::registration::CorrespondenceRejectorSampleConsensus<Point> rejector;
    rejector.setInputSource(source);
    rejector.setInputTarget(target);
    rejector.setInlierThreshold(inlier_threshold);
    rejector.setMaximumIterations(max_iterations);
    rejector.setRefineModel(refine);
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
    copyMatrix(rejector.getBestTransformation(), out_matrix16);
    return !remaining.empty();
}

inline void rejectSurfaceNormal(const CloudPtr& source, const CloudPtr& target,
                                const NormalCloudPtr& source_normals,
                                const NormalCloudPtr& target_normals,
                                const std::vector<int>& pairs,
                                const std::vector<float>& distances,
                                double threshold, std::vector<int>& out_pairs,
                                std::vector<float>& out_distances) {
    pcl::registration::CorrespondenceRejectorSurfaceNormal rejector;
    rejector.initializeDataContainer<Point, pcl::Normal>();
    rejector.setInputSource<Point>(source);
    rejector.setInputTarget<Point>(target);
    rejector.setInputNormals<Point, pcl::Normal>(source_normals);
    rejector.setTargetNormals<Point, pcl::Normal>(target_normals);
    rejector.setThreshold(threshold);
    pcl::Correspondences remaining;
    rejector.getRemainingCorrespondences(toCorrespondences(pairs, distances),
                                        remaining);
    fromCorrespondences(remaining, out_pairs, out_distances);
}

inline bool estimateTransformation(int method, const CloudPtr& source,
                                   const CloudPtr& target,
                                   const std::vector<int>& pairs,
                                   float* out_matrix16) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    const std::vector<float> no_distances;
    if (method == 0) {
        pcl::registration::TransformationEstimationSVD<Point, Point> svd;
        if (pairs.empty()) {
            svd.estimateRigidTransformation(*source, *target, m);
        } else {
            svd.estimateRigidTransformation(
                *source, *target, toCorrespondences(pairs, no_distances), m);
        }
    } else if (method == 1) {
        pcl::registration::TransformationEstimationLM<Point, Point> lm;
        if (pairs.empty()) {
            lm.estimateRigidTransformation(*source, *target, m);
        } else {
            lm.estimateRigidTransformation(
                *source, *target, toCorrespondences(pairs, no_distances), m);
        }
    } else {
        return false;
    }
    copyMatrix(m, out_matrix16);
    return true;
}

// TransformationValidationEuclidean also has a `threshold`, but only
// isValid() reads it (score < threshold); the score itself ignores it,
// so the wrapper leaves that comparison to the caller.
inline double validateTransformationEuclidean(const CloudPtr& source,
                                              const CloudPtr& target,
                                              const float* matrix16,
                                              double max_range) {
    pcl::registration::TransformationValidationEuclidean<Point, Point> validation;
    if (max_range > 0) {
        validation.setMaxRange(max_range);
    }
    return validation.validateTransformation(source, target,
                                             readMatrix(matrix16));
}

}  // namespace pclcompat
