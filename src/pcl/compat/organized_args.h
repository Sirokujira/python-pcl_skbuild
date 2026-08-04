// Reach IntegralImageNormalEstimation's nested enum, and give
// ConditionalEuclideanClustering a condition Cython can supply.
//
// Two more of the same two shapes the other shims handle:
//
//  * `setNormalEstimationMethod` takes an enum nested inside a class
//    TEMPLATE. Cython has no syntax for naming one (`Klass[A, B].MEMBER`
//    is rejected) and C++ will not implicitly convert an int, so neither
//    side can bridge it alone — same as HarrisKeypoint3D's
//    ResponseMethod (see keypoint_args.h).
//  * `setConditionFunction` takes a `std::function`, which cannot be
//    built from a Python callable — same as the grabber callback (see
//    grabber_callback.h). The shim takes a C function pointer plus an
//    opaque user_data and wraps them.
//
// The clustering condition runs on PCL's calling thread with whatever
// GIL state the caller had, so the Cython side must acquire it, exactly
// as the grabber trampoline does.
#pragma once

#include <functional>

#include <pcl/features/integral_image_normal.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using IntegralImageNormal =
    pcl::IntegralImageNormalEstimation<Point, pcl::Normal>;

const int IINE_COVARIANCE_MATRIX = IntegralImageNormal::COVARIANCE_MATRIX;
const int IINE_AVERAGE_3D_GRADIENT = IntegralImageNormal::AVERAGE_3D_GRADIENT;
const int IINE_AVERAGE_DEPTH_CHANGE =
    IntegralImageNormal::AVERAGE_DEPTH_CHANGE;
const int IINE_SIMPLE_3D_GRADIENT = IntegralImageNormal::SIMPLE_3D_GRADIENT;

inline void setNormalEstimationMethod(IntegralImageNormal& estimator,
                                      int method) {
    estimator.setNormalEstimationMethod(
        static_cast<IntegralImageNormal::NormalEstimationMethod>(method));
}

/// Cython supplies this: a `cdef ... noexcept nogil` function deciding
/// whether two neighbouring points belong to the same cluster.
typedef bool (*ClusterConditionFn)(const Point& a, const Point& b,
                                   float squared_distance, void* user_data);

inline void setClusterCondition(
        pcl::ConditionalEuclideanClustering<Point>& clustering,
        ClusterConditionFn fn, void* user_data) {
    clustering.setConditionFunction(
        [fn, user_data](const Point& a, const Point& b, float sqr_distance) {
            return fn(a, b, sqr_distance, user_data);
        });
}

}  // namespace pclcompat
