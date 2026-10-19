// Search helpers Cython cannot state directly: computeCameraMatrix fills
// an Eigen::Matrix3f, which crosses as a row-major float[9].
#pragma once

#include <pcl/point_types.h>
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

}  // namespace pclcompat
