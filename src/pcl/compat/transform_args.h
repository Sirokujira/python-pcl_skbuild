// Apply a 4x4 transform to a cloud.
//
// Every registration algorithm and every recognized instance hands back
// an Eigen::Matrix4f, and until this shim existed there was no way to
// APPLY one: a caller had to round-trip the cloud through numpy, which
// costs the per-point Python access this binding otherwise avoids
// (~85 ns/point, see bench/README.md).
//
// The matrix arrives as 16 floats in COLUMN-MAJOR order — Eigen's own
// storage, which is exactly what `finalTransformation` and the grouping
// results already emit, so a result can be fed straight back in.
//
// One overload per wrapped point type rather than a template: Cython
// calls these directly, and overload resolution on reference types is
// something it can express.
#pragma once

#include <cmath>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

inline Eigen::Matrix4f asMatrix(const float* column_major16) {
    Eigen::Matrix4f m;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            m(row, col) = column_major16[col * 4 + row];
        }
    }
    return m;
}

template <typename PointT>
inline void transformWith(const pcl::PointCloud<PointT>& in,
                          pcl::PointCloud<PointT>& out,
                          const float* matrix16) {
    pcl::transformPointCloud(in, out, asMatrix(matrix16));
}

inline void transformCloud(const pcl::PointCloud<pcl::PointXYZ>& in,
                           pcl::PointCloud<pcl::PointXYZ>& out,
                           const float* matrix16) {
    transformWith(in, out, matrix16);
}

inline void transformCloud(const pcl::PointCloud<pcl::PointXYZI>& in,
                           pcl::PointCloud<pcl::PointXYZI>& out,
                           const float* matrix16) {
    transformWith(in, out, matrix16);
}

inline void transformCloud(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                           pcl::PointCloud<pcl::PointXYZRGB>& out,
                           const float* matrix16) {
    transformWith(in, out, matrix16);
}

inline void transformCloud(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                           pcl::PointCloud<pcl::PointXYZRGBA>& out,
                           const float* matrix16) {
    transformWith(in, out, matrix16);
}

/// True when the upper-left 3x3 is a rotation (orthonormal, det +1) and
/// the bottom row is (0, 0, 0, 1) — i.e. the matrix is a rigid motion.
///
/// PCL's transformPointCloud applies whatever it is given without
/// complaint, so a caller who passes a scaling or reflecting matrix by
/// mistake gets a silently distorted cloud. The wrapper warns off that
/// case rather than PCL discovering it later.
inline bool isRigidTransform(const float* matrix16, float tolerance) {
    const Eigen::Matrix4f m = asMatrix(matrix16);
    const Eigen::Matrix3f r = m.topLeftCorner<3, 3>();

    if (std::abs(m(3, 0)) > tolerance || std::abs(m(3, 1)) > tolerance ||
        std::abs(m(3, 2)) > tolerance ||
        std::abs(m(3, 3) - 1.0f) > tolerance) {
        return false;
    }
    const Eigen::Matrix3f should_be_identity = r.transpose() * r;
    if (!should_be_identity.isApprox(Eigen::Matrix3f::Identity(), tolerance)) {
        return false;
    }
    return std::abs(r.determinant() - 1.0f) <= tolerance;
}

}  // namespace pclcompat
