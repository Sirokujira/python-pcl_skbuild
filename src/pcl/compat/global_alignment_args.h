// Global registration: align two clouds with NO initial guess.
//
// Every registration algorithm wrapped so far — ICP, ICP-NL, GICP, NDT —
// refines a pose that is already roughly right; point them at clouds
// metres and radians apart and they converge on nonsense. PCL's answer
// is SampleConsensusPrerejective: sample correspondences from FPFH
// features, reject the geometrically impossible ones with a polygon
// test, and keep the pose with the best inlier fraction.
//
// The shim exists because the algorithm's inputs and outputs are all
// things a pxd cannot state: feature clouds of a fixed-size descriptor
// struct, an Eigen::Matrix4f result, and a pcl::Indices inlier list. It
// takes descriptors as the flat float arrays FPFHEstimation already
// hands back — the same currency matching_args.h deals in — so nothing
// has to round-trip through a feature-cloud type Python never sees.
#pragma once

#include <cstddef>
#include <vector>

#include <pcl/features/fpfh.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/sample_consensus_prerejective.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using Features = pcl::PointCloud<pcl::FPFHSignature33>;

inline Features::Ptr featureCloud(const std::vector<float>& flat) {
    Features::Ptr features(new Features());
    const std::size_t n = flat.size() / 33;
    features->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (int j = 0; j < 33; ++j) {
            (*features)[i].histogram[j] = flat[i * 33 + j];
        }
    }
    return features;
}

/// Aligns *source* onto *target*. Writes the pose into `out_matrix16`
/// (column-major, Eigen's own storage) and the inlier indices — into
/// the SOURCE cloud — into `out_inliers`. Returns whether it converged;
/// `out_fitness` gets the mean squared distance to the target.
///
/// `max_correspondence_distance` is the Euclidean threshold below which
/// a transformed source point counts as an inlier, and
/// `inlier_fraction` how many must qualify before a pose is accepted at
/// all — the two settings that decide whether this finds anything.
inline bool globalAlignment(const std::shared_ptr<Cloud>& source,
                            const std::vector<float>& source_features,
                            const std::shared_ptr<Cloud>& target,
                            const std::vector<float>& target_features,
                            int max_iterations, int number_of_samples,
                            int correspondence_randomness,
                            float similarity_threshold,
                            float max_correspondence_distance,
                            float inlier_fraction,
                            Cloud& out_aligned,
                            float* out_matrix16,
                            std::vector<int>& out_inliers,
                            double* out_fitness) {
    pcl::SampleConsensusPrerejective<Point, Point, pcl::FPFHSignature33> align;
    align.setInputSource(source);
    align.setSourceFeatures(featureCloud(source_features));
    align.setInputTarget(target);
    align.setTargetFeatures(featureCloud(target_features));
    align.setMaximumIterations(max_iterations);
    align.setNumberOfSamples(number_of_samples);
    align.setCorrespondenceRandomness(correspondence_randomness);
    align.setSimilarityThreshold(similarity_threshold);
    align.setMaxCorrespondenceDistance(max_correspondence_distance);
    align.setInlierFraction(inlier_fraction);

    align.align(out_aligned);

    const Eigen::Matrix4f m = align.getFinalTransformation();
    const float* data = m.data();
    for (int i = 0; i < 16; ++i) {
        out_matrix16[i] = data[i];
    }

    out_inliers.clear();
    for (const auto index : align.getInliers()) {
        out_inliers.push_back(static_cast<int>(index));
    }

    // getFitnessScore() searches the target tree, so it is only
    // meaningful once something converged; PCL returns a huge number
    // otherwise and the wrapper passes that through unchanged.
    *out_fitness = align.getFitnessScore();
    return align.hasConverged();
}

}  // namespace pclcompat
