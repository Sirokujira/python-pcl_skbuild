// Descriptor matching: for every scene descriptor, the nearest model
// descriptor — the step between FPFH/SHOT estimation and correspondence
// grouping.
//
// The pyx layer holds descriptors as numpy arrays, and PCL's matcher is
// KdTreeFLANN<Descriptor> over a PointCloud of a fixed-size struct. The
// shim owns the round trip: flat floats in, a FLANN tree per call, index
// pairs and distances out. Cython never sees the descriptor cloud, the
// tree, or the per-type template instantiation.
//
// Direction follows PCL's correspondence-grouping tutorial: the tree is
// built over the MODEL and queried with every SCENE descriptor, so each
// scene point contributes at most one correspondence.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

template <typename Desc, int Dim>
inline int matchWith(const std::vector<float>& model,
                     const std::vector<float>& scene,
                     float max_distance,
                     std::vector<int>& out_model_indices,
                     std::vector<int>& out_scene_indices,
                     std::vector<float>& out_distances) {
    out_model_indices.clear();
    out_scene_indices.clear();
    out_distances.clear();

    const std::size_t n_model = model.size() / Dim;
    const std::size_t n_scene = scene.size() / Dim;
    if (n_model == 0 || n_scene == 0) {
        return 0;
    }

    // NaN rows — SHOT's "no descriptor here" — may occur on the model
    // side too, and FLANN would happily index them. Keep only finite
    // rows and remember which original row each tree entry was.
    typename pcl::PointCloud<Desc>::Ptr cloud(new pcl::PointCloud<Desc>());
    std::vector<int> original_row;
    cloud->reserve(n_model);
    original_row.reserve(n_model);
    for (std::size_t i = 0; i < n_model; ++i) {
        bool finite = true;
        for (int j = 0; j < Dim; ++j) {
            if (!std::isfinite(model[i * Dim + j])) {
                finite = false;
                break;
            }
        }
        if (!finite) {
            continue;
        }
        Desc entry;
        float* target = reinterpret_cast<float*>(&entry);
        for (int j = 0; j < Dim; ++j) {
            target[j] = model[i * Dim + j];
        }
        cloud->push_back(entry);
        original_row.push_back(static_cast<int>(i));
    }
    if (cloud->empty()) {
        return 0;
    }

    pcl::KdTreeFLANN<Desc> tree;
    tree.setInputCloud(cloud);

    const float max_d2 =
        max_distance < 0 ? -1.0f : max_distance * max_distance;
    std::vector<int> index(1);
    std::vector<float> squared(1);
    Desc query;
    for (std::size_t i = 0; i < n_scene; ++i) {
        float* target = reinterpret_cast<float*>(&query);
        bool finite = true;
        for (int j = 0; j < Dim; ++j) {
            target[j] = scene[i * Dim + j];
            if (!std::isfinite(target[j])) {
                finite = false;
            }
        }
        // A NaN descriptor is SHOT's "no descriptor here"; FLANN would
        // return garbage for it rather than failing.
        if (!finite) {
            continue;
        }
        if (tree.nearestKSearch(query, 1, index, squared) != 1) {
            continue;
        }
        if (max_d2 >= 0 && squared[0] > max_d2) {
            continue;
        }
        out_model_indices.push_back(original_row[index[0]]);
        out_scene_indices.push_back(static_cast<int>(i));
        out_distances.push_back(std::sqrt(squared[0]));
    }
    return static_cast<int>(out_scene_indices.size());
}

/// *model* and *scene* are flat row-major descriptor arrays, 33 floats
/// per row. `max_distance` < 0 means unbounded. Returns how many
/// correspondences were kept.
inline int matchFpfhDescriptors(const std::vector<float>& model,
                                const std::vector<float>& scene,
                                float max_distance,
                                std::vector<int>& out_model_indices,
                                std::vector<int>& out_scene_indices,
                                std::vector<float>& out_distances) {
    return matchWith<pcl::FPFHSignature33, 33>(
        model, scene, max_distance, out_model_indices, out_scene_indices,
        out_distances);
}

/// The same for SHOT, 352 floats per row. Scene rows containing NaN —
/// SHOT's "no descriptor here" — are skipped.
inline int matchShotDescriptors(const std::vector<float>& model,
                                const std::vector<float>& scene,
                                float max_distance,
                                std::vector<int>& out_model_indices,
                                std::vector<int>& out_scene_indices,
                                std::vector<float>& out_distances) {
    return matchWith<pcl::SHOT352, 352>(
        model, scene, max_distance, out_model_indices, out_scene_indices,
        out_distances);
}

}  // namespace pclcompat
