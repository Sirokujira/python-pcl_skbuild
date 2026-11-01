// Search helpers Cython cannot state directly: computeCameraMatrix fills
// an Eigen::Matrix3f, which crosses as a row-major float[9]; FlannSearch
// picks its index through creator classes nested in the template.
#pragma once

#include <pcl/point_types.h>
#include <stdexcept>

#include <pcl/search/flann_search.h>
#include <pcl/search/organized.h>

namespace pclcompat {

inline void organizedCameraMatrix(
    const pcl::search::OrganizedNeighbor<pcl::PointXYZ>& search,
    float* out_matrix9) {
    Eigen::Matrix3f m;
    search.computeCameraMatrix(m);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out_matrix9[3 * r + c] = m(r, c);
        }
    }
}

const int FLANN_INDEX_KDTREE_SINGLE = 0;  // one kd-tree (PCL's default)
const int FLANN_INDEX_KDTREE_MULTI = 1;   // `trees` randomized kd-trees
const int FLANN_INDEX_KMEANS = 2;         // hierarchical k-means tree

/// A FlannSearch over the chosen index; the caller owns it.
inline pcl::search::FlannSearch<pcl::PointXYZ>* newFlannSearch(int index,
                                                              int trees,
                                                              bool sorted) {
    using Search = pcl::search::FlannSearch<pcl::PointXYZ>;
    Search::FlannIndexCreatorPtr creator;
    switch (index) {
    case FLANN_INDEX_KDTREE_SINGLE:
        creator.reset(new Search::KdTreeIndexCreator());
        break;
    case FLANN_INDEX_KDTREE_MULTI:
        creator.reset(new Search::KdTreeMultiIndexCreator(trees));
        break;
    case FLANN_INDEX_KMEANS:
        creator.reset(new Search::KMeansIndexCreator());
        break;
    default:
        throw std::invalid_argument("unknown FLANN index type");
    }
    return new Search(sorted, creator);
}

}  // namespace pclcompat
