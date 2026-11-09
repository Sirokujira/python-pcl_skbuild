// Hypothesis verification (pcl/recognition/hv/) behind three functions.
//
// Every verifier takes its models as a non-const
// `std::vector<PointCloud<ModelT>::ConstPtr>&` -- a vector of shared_ptr
// to const, which a vector of the shared_ptr Cython holds does not bind
// to -- and reports through `getMask(std::vector<bool>&)`. Each function
// here takes the scene and the models already placed in it, runs one
// verifier, and returns the mask as ints.
//
// Two things the PCL API leaves to the caller and the wrappers would
// otherwise get silently wrong:
// - PapazovHV iterates `complete_models_`, which addModels() never fills
//   unless occlusion reasoning is on; with only addModels() it returns an
//   EMPTY mask (measured in plain C++ against PCL 1.14;
//   GreedyVerification reads the visible models and is unaffected).
//   Without occlusion reasoning the complete models are the visible ones,
//   so both are always set.
// - addModels() warns on stderr unless an occlusion cloud was set; the
//   scene is what it falls back to, so it is set explicitly.
//
// GlobalHypothesesVerification is deliberately absent: see the README's
// "Not wrapped, and why".
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/recognition/hv/hv_papazov.h>

#include "pcl/compat/hv_go_fixed.h"

namespace pclcompat {

using HVPoint = pcl::PointXYZ;
using HVCloud = pcl::PointCloud<HVPoint>;
using HVCloudPtr = std::shared_ptr<HVCloud>;

template <typename Verifier>
inline void runVerifier(Verifier& hv, const HVCloudPtr& scene,
                        const std::vector<HVCloudPtr>& models,
                        float resolution, float inlier_threshold,
                        float occlusion_threshold, std::vector<int>& out_mask) {
    hv.setResolution(resolution);
    hv.setInlierThreshold(inlier_threshold);
    hv.setOcclusionThreshold(occlusion_threshold);
    hv.setSceneCloud(scene);
    hv.setOcclusionCloud(scene);

    std::vector<HVCloud::ConstPtr> const_models(models.begin(), models.end());
    hv.addModels(const_models, false);
    hv.addCompleteModels(const_models);
    hv.verify();

    std::vector<bool> mask;
    hv.getMask(mask);
    out_mask.assign(mask.begin(), mask.end());
}

inline void greedyVerification(const HVCloudPtr& scene,
                               const std::vector<HVCloudPtr>& models,
                               float resolution, float inlier_threshold,
                               float occlusion_threshold, float lambda,
                               std::vector<int>& out_mask) {
    pcl::GreedyVerification<HVPoint, HVPoint> hv(lambda);
    runVerifier(hv, scene, models, resolution, inlier_threshold,
                occlusion_threshold, out_mask);
}

inline void papazovVerification(const HVCloudPtr& scene,
                                const std::vector<HVCloudPtr>& models,
                                float resolution, float inlier_threshold,
                                float occlusion_threshold,
                                float conflict_threshold,
                                float support_threshold,
                                float penalty_threshold,
                                std::vector<int>& out_mask) {
    pcl::PapazovHV<HVPoint, HVPoint> hv;
    hv.setConflictThreshold(conflict_threshold);
    hv.setSupportThreshold(support_threshold);
    hv.setPenaltyThreshold(penalty_threshold);
    runVerifier(hv, scene, models, resolution, inlier_threshold,
                occlusion_threshold, out_mask);
}

/// GlobalHypothesesVerification, from the corrected copy in
/// hv_go_fixed.h (PCL's keeps only the model points whose normals are NOT
/// finite).
inline void globalVerification(const HVCloudPtr& scene,
                               const std::vector<HVCloudPtr>& models,
                               float resolution, float inlier_threshold,
                               float occlusion_threshold, float regularizer,
                               float radius_clutter, float clutter_regularizer,
                               float radius_normals, bool detect_clutter,
                               int max_iterations, float initial_temperature,
                               float occupancy_grid_resolution,
                               std::vector<int>& out_mask) {
    pcl::pclcompat_hv::GlobalHypothesesVerification<HVPoint, HVPoint> hv;
    hv.setRegularizer(regularizer);
    hv.setRadiusClutter(radius_clutter);
    hv.setClutterRegularizer(clutter_regularizer);
    hv.setRadiusNormals(radius_normals);
    hv.setDetectClutter(detect_clutter);
    hv.setMaxIterations(max_iterations);
    hv.setInitialTemp(initial_temperature);
    hv.setResolutionOccupancyGrid(occupancy_grid_resolution);
    runVerifier(hv, scene, models, resolution, inlier_threshold,
                occlusion_threshold, out_mask);
}

}  // namespace pclcompat
