// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/keypoint_args.h.
#pragma once

#include <pcl/keypoints/harris_3d.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int HARRIS_METHOD_HARRIS = 1;
const int HARRIS_METHOD_NOBLE = 2;
const int HARRIS_METHOD_LOWE = 3;
const int HARRIS_METHOD_TOMASI = 4;
const int HARRIS_METHOD_CURVATURE = 5;

void setHarrisMethod(
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>& detector,
    int method);

}  // namespace pclcompat
