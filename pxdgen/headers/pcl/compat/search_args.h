// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/search_args.h.
#pragma once

#include <pcl/point_types.h>
#include <pcl/search/flann_search.h>
#include <pcl/search/organized.h>

namespace pclcompat {

// The camera matrix OrganizedNeighbor estimated, row-major 3x3.
void organizedCameraMatrix(
    const pcl::search::OrganizedNeighbor<pcl::PointXYZ>& search,
    float* out_matrix9);

const int FLANN_INDEX_KDTREE_SINGLE = 0;
const int FLANN_INDEX_KDTREE_MULTI = 1;
const int FLANN_INDEX_KMEANS = 2;

// FlannSearch over one of the index types above (`trees` for the
// multi-tree one); the caller owns the result.
pcl::search::FlannSearch<pcl::PointXYZ>* newFlannSearch(int index, int trees,
                                                       bool sorted);

}  // namespace pclcompat
