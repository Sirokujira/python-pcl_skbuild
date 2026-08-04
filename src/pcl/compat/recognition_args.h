// Correspondence grouping: find instances of a model inside a scene.
//
// Everything about this API is Cython-inexpressible at once:
//
//  * correspondences travel as `pcl::CorrespondencesPtr`, a shared_ptr to
//    a vector of a struct whose `distance` member lives in an anonymous
//    union;
//  * results come back as `std::vector<Eigen::Matrix4f,
//    Eigen::aligned_allocator<Eigen::Matrix4f>>` — an Eigen type in an
//    Eigen allocator, which is not the same C++ type as a plain vector;
//  * the clustered correspondences are a vector of *those* vectors.
//
// Rather than mirror four types, this shim owns the pipeline: indices in,
// flat float transforms plus flat cluster indices out. Both grouping
// algorithms PCL ships get the same shape, so the wrapper treats them
// alike.
#pragma once

#include <cstddef>
#include <vector>

#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using RefCloud = pcl::PointCloud<pcl::ReferenceFrame>;

/// Copies `transforms` out as 16 floats per instance, column-major (what
/// Eigen stores natively and what numpy calls Fortran order), and the
/// clustered correspondences as a flat (model, scene) index pair list
/// plus a per-instance count so the caller can split it back up.
inline void flattenInstances(
        const std::vector<Eigen::Matrix4f,
                          Eigen::aligned_allocator<Eigen::Matrix4f>>& transforms,
        const std::vector<pcl::Correspondences>& clustered,
        std::vector<float>& out_transforms,
        std::vector<int>& out_corr_counts,
        std::vector<int>& out_corr_pairs) {
    out_transforms.clear();
    out_corr_counts.clear();
    out_corr_pairs.clear();

    for (const auto& m : transforms) {
        const float* data = m.data();
        out_transforms.insert(out_transforms.end(), data, data + 16);
    }
    for (const auto& corrs : clustered) {
        out_corr_counts.push_back(static_cast<int>(corrs.size()));
        for (const auto& c : corrs) {
            out_corr_pairs.push_back(static_cast<int>(c.index_query));
            out_corr_pairs.push_back(static_cast<int>(c.index_match));
        }
    }
}

/// Builds the correspondence list PCL wants from two parallel index
/// arrays — the form a caller already has after matching descriptors.
inline pcl::CorrespondencesPtr makeCorrespondences(
        const std::vector<int>& model_indices,
        const std::vector<int>& scene_indices,
        const std::vector<float>& distances) {
    pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
    corrs->reserve(model_indices.size());
    for (std::size_t i = 0; i < model_indices.size(); ++i) {
        corrs->push_back(pcl::Correspondence(
            model_indices[i], scene_indices[i],
            i < distances.size() ? distances[i] : 0.0f));
    }
    return corrs;
}

// --- geometric consistency -------------------------------------------

/// Returns the number of model instances found. `gc_size` is the
/// consensus set resolution in metric units and `gc_threshold` the
/// smallest cluster that counts as an instance (3 is PCL's floor: a
/// 6-DOF pose needs three correspondences).
inline int geometricConsistencyGrouping(
        const std::shared_ptr<Cloud>& model,
        const std::shared_ptr<Cloud>& scene,
        const std::vector<int>& model_indices,
        const std::vector<int>& scene_indices,
        const std::vector<float>& distances,
        double gc_size, int gc_threshold,
        std::vector<float>& out_transforms,
        std::vector<int>& out_corr_counts,
        std::vector<int>& out_corr_pairs) {
    pcl::GeometricConsistencyGrouping<Point, Point> gc;
    gc.setGCSize(gc_size);
    gc.setGCThreshold(gc_threshold);
    gc.setInputCloud(model);
    gc.setSceneCloud(scene);
    gc.setModelSceneCorrespondences(
        makeCorrespondences(model_indices, scene_indices, distances));

    std::vector<Eigen::Matrix4f,
                Eigen::aligned_allocator<Eigen::Matrix4f>> transforms;
    std::vector<pcl::Correspondences> clustered;
    gc.recognize(transforms, clustered);

    flattenInstances(transforms, clustered, out_transforms, out_corr_counts,
                     out_corr_pairs);
    return static_cast<int>(transforms.size());
}

// --- Hough voting -----------------------------------------------------

/// Local reference frames for `cloud`, which Hough3DGrouping votes in.
/// BOARD is the estimator every PCL recognition tutorial uses; it needs
/// normals, so the shim computes those too rather than making the caller
/// carry a second cloud through an Eigen-typed API.
inline RefCloud::Ptr boardReferenceFrames(const std::shared_ptr<Cloud>& cloud,
                                          float rf_radius);

/// Same contract as geometricConsistencyGrouping, voting in a Hough
/// space of `bin_size` instead. `rf_radius` is the support radius of the
/// local reference frames.
inline int hough3DGrouping(const std::shared_ptr<Cloud>& model,
                           const std::shared_ptr<Cloud>& scene,
                           const std::vector<int>& model_indices,
                           const std::vector<int>& scene_indices,
                           const std::vector<float>& distances,
                           float rf_radius, double bin_size,
                           double threshold,
                           std::vector<float>& out_transforms,
                           std::vector<int>& out_corr_counts,
                           std::vector<int>& out_corr_pairs) {
    pcl::Hough3DGrouping<Point, Point, pcl::ReferenceFrame,
                         pcl::ReferenceFrame> hough;
    hough.setHoughBinSize(bin_size);
    hough.setHoughThreshold(threshold);
    hough.setUseInterpolation(true);
    hough.setUseDistanceWeight(false);
    hough.setInputCloud(model);
    hough.setInputRf(boardReferenceFrames(model, rf_radius));
    hough.setSceneCloud(scene);
    hough.setSceneRf(boardReferenceFrames(scene, rf_radius));
    hough.setModelSceneCorrespondences(
        makeCorrespondences(model_indices, scene_indices, distances));

    std::vector<Eigen::Matrix4f,
                Eigen::aligned_allocator<Eigen::Matrix4f>> transforms;
    std::vector<pcl::Correspondences> clustered;
    hough.recognize(transforms, clustered);

    flattenInstances(transforms, clustered, out_transforms, out_corr_counts,
                     out_corr_pairs);
    return static_cast<int>(transforms.size());
}

}  // namespace pclcompat

// Defined after the users above so the header reads top-down.
#include <pcl/features/normal_3d.h>

namespace pclcompat {

inline RefCloud::Ptr boardReferenceFrames(const std::shared_ptr<Cloud>& cloud,
                                          float rf_radius) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<Point, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(rf_radius);
    ne.compute(*normals);

    RefCloud::Ptr frames(new RefCloud());
    pcl::BOARDLocalReferenceFrameEstimation<Point, pcl::Normal,
                                            pcl::ReferenceFrame> rf;
    rf.setFindHoles(true);
    rf.setRadiusSearch(rf_radius);
    rf.setInputCloud(cloud);
    rf.setInputNormals(normals);
    rf.compute(*frames);
    return frames;
}

}  // namespace pclcompat
