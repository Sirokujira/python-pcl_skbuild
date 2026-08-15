// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/matching_args.h (reached as
// "pcl/compat/matching_args.h" because CMake adds src/ to the includes).
//
// The real header builds a pcl::KdTreeFLANN over a descriptor cloud;
// this one names only the two entry points Cython calls, so pxd
// generation still runs without PCL or FLANN installed.
#pragma once

#include <vector>

namespace pclcompat {

int matchFpfhDescriptors(const std::vector<float>& model,
                         const std::vector<float>& scene,
                         float max_distance,
                         std::vector<int>& out_model_indices,
                         std::vector<int>& out_scene_indices,
                         std::vector<float>& out_distances);

int matchShotDescriptors(const std::vector<float>& model,
                         const std::vector<float>& scene,
                         float max_distance,
                         std::vector<int>& out_model_indices,
                         std::vector<int>& out_scene_indices,
                         std::vector<float>& out_distances);

}  // namespace pclcompat
