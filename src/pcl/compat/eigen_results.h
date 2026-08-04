// Flatten PCL's Eigen-typed results into plain float buffers.
//
// The rung-4 C++ shim pattern again (see grabber_callback.h): a mirror
// header may not name Eigen types -- it has to stay self-contained so pxd
// generation runs without Eigen installed -- and Cython would gain
// nothing from an Eigen pxd here anyway, because every one of these
// results ends up in a numpy array.
//
// So each function copies its Eigen result into a caller-provided
// `float*`, and the wrapper hands that buffer to numpy. The buffer sizes
// are part of each function's contract and are asserted on the Cython
// side by construction (fixed-size arrays).
//
// One overload per concrete type rather than a template, for the same
// reason as grabber_callback.h: Cython calls it directly, and overload
// resolution on pointer types is something it can express.
#pragma once

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ndt.h>

#include <vector>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;

// --- registration: Eigen::Matrix4f -> float[16] ----------------------
//
// Eigen is column-major by default, so the copy is already the Fortran
// ordering numpy wants for a transformation matrix.
template <typename RegT>
inline void copyTransform(RegT& reg, float* out16) {
    const Eigen::Matrix4f m = reg.getFinalTransformation();
    const float* data = m.data();
    for (int i = 0; i < 16; ++i) {
        out16[i] = data[i];
    }
}

inline void finalTransformation(pcl::IterativeClosestPoint<Point, Point>& reg,
                                float* out16) {
    copyTransform(reg, out16);
}

inline void finalTransformation(
        pcl::IterativeClosestPointNonLinear<Point, Point>& reg, float* out16) {
    copyTransform(reg, out16);
}

inline void finalTransformation(
        pcl::GeneralizedIterativeClosestPoint<Point, Point>& reg,
        float* out16) {
    copyTransform(reg, out16);
}

inline void finalTransformation(
        pcl::NormalDistributionsTransform<Point, Point>& reg, float* out16) {
    copyTransform(reg, out16);
}

// --- moment of inertia: Eigen vectors/matrices -> float buffers ------

inline void massCenter(pcl::MomentOfInertiaEstimation<Point>& est,
                       float* out3) {
    Eigen::Vector3f center;
    est.getMassCenter(center);
    for (int i = 0; i < 3; ++i) {
        out3[i] = center[i];
    }
}

/// min x,y,z then max x,y,z.
inline void axisAlignedBoundingBox(pcl::MomentOfInertiaEstimation<Point>& est,
                                   float* out6) {
    Point min_point, max_point;
    est.getAABB(min_point, max_point);
    out6[0] = min_point.x; out6[1] = min_point.y; out6[2] = min_point.z;
    out6[3] = max_point.x; out6[4] = max_point.y; out6[5] = max_point.z;
}

/// min x,y,z | max x,y,z | position x,y,z | 3x3 rotation, row-major.
inline void orientedBoundingBox(pcl::MomentOfInertiaEstimation<Point>& est,
                                float* out18) {
    Point min_point, max_point, position;
    Eigen::Matrix3f rotation;
    est.getOBB(min_point, max_point, position, rotation);
    out18[0] = min_point.x; out18[1] = min_point.y; out18[2] = min_point.z;
    out18[3] = max_point.x; out18[4] = max_point.y; out18[5] = max_point.z;
    out18[6] = position.x;  out18[7] = position.y;  out18[8] = position.z;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out18[9 + r * 3 + c] = rotation(r, c);
        }
    }
}

/// major, middle, minor.
inline void eigenValues(pcl::MomentOfInertiaEstimation<Point>& est,
                        float* out3) {
    est.getEigenValues(out3[0], out3[1], out3[2]);
}

/// major x,y,z | middle x,y,z | minor x,y,z.
inline void eigenVectors(pcl::MomentOfInertiaEstimation<Point>& est,
                         float* out9) {
    Eigen::Vector3f major, middle, minor;
    est.getEigenVectors(major, middle, minor);
    for (int i = 0; i < 3; ++i) {
        out9[i] = major[i];
        out9[3 + i] = middle[i];
        out9[6 + i] = minor[i];
    }
}

// --- octree: Eigen-aligned vector -> plain std::vector ---------------
//
// getOccupiedVoxelCenters fills a std::vector<PointT,
// Eigen::aligned_allocator<PointT>>. That is a DIFFERENT C++ type from
// std::vector<PointT>, so Cython cannot pass one for the other; the copy
// here is what makes the result reachable at all.
inline int occupiedVoxelCenters(
        pcl::octree::OctreePointCloudSearch<Point>& octree,
        std::vector<Point>& out) {
    typename pcl::octree::OctreePointCloudSearch<Point>::AlignedPointTVector
        centers;
    const int count = static_cast<int>(octree.getOccupiedVoxelCenters(centers));
    out.assign(centers.begin(), centers.end());
    return count;
}

}  // namespace pclcompat
