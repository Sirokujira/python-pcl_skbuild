// Reach HarrisKeypoint3D's nested ResponseMethod enum from Cython.
//
// `setMethod` takes `HarrisKeypoint3D<PointInT, PointOutT>::ResponseMethod`
// — an enum nested inside a class TEMPLATE. Cython has no syntax for
// naming that (`Klass[A, B].MEMBER` is rejected), and C++ will not
// implicitly convert an int to it, so neither side can bridge the gap
// alone.
//
// The shim takes an int and does the cast, and re-exports the enumerators
// as namespace-scope constants so their values come from the header
// rather than being copied into Python and left to drift.
#pragma once

#include <pcl/keypoints/harris_3d.h>
#include <pcl/point_types.h>

namespace pclcompat {

using HarrisXYZI = pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;

const int HARRIS_METHOD_HARRIS = HarrisXYZI::HARRIS;
const int HARRIS_METHOD_NOBLE = HarrisXYZI::NOBLE;
const int HARRIS_METHOD_LOWE = HarrisXYZI::LOWE;
const int HARRIS_METHOD_TOMASI = HarrisXYZI::TOMASI;
const int HARRIS_METHOD_CURVATURE = HarrisXYZI::CURVATURE;

inline void setHarrisMethod(HarrisXYZI& detector, int method) {
    detector.setMethod(static_cast<HarrisXYZI::ResponseMethod>(method));
}

}  // namespace pclcompat
