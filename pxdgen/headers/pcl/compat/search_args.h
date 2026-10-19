// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/search_args.h.
#pragma once

#include <pcl/point_types.h>
#include <pcl/search/organized.h>

namespace pclcompat {

// The camera matrix OrganizedNeighbor estimated, row-major 3x3.
void organizedCameraMatrix(
    const pcl::search::OrganizedNeighbor<pcl::PointXYZ>& search,
    float* out_matrix9);

}  // namespace pclcompat
