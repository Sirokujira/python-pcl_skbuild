// Take Eigen-typed ARGUMENTS as plain floats.
//
// The mirror of eigen_results.h: that one copies Eigen values OUT of PCL,
// this one passes them IN. Same reason either way — a mirror header may
// not name Eigen (it has to stay self-contained so pxd generation runs
// without Eigen installed), and Cython has no way to build an
// Eigen::Vector4f.
//
// One overload per concrete type rather than a template, so Cython can
// call it directly.
#pragma once

#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Point = pcl::PointXYZ;

/// The box corners. The fourth component of PCL's Vector4f is unused for
/// a position, and passing 1.0 is what every PCL example does.
inline void setCropBoxMin(pcl::CropBox<Point>& box,
                          float x, float y, float z) {
    box.setMin(Eigen::Vector4f(x, y, z, 1.0f));
}

inline void setCropBoxMax(pcl::CropBox<Point>& box,
                          float x, float y, float z) {
    box.setMax(Eigen::Vector4f(x, y, z, 1.0f));
}

inline void setCropBoxTranslation(pcl::CropBox<Point>& box,
                                  float x, float y, float z) {
    box.setTranslation(Eigen::Vector3f(x, y, z));
}

/// Rotation as roll/pitch/yaw in radians, which is what PCL's Vector3f
/// overload means here.
inline void setCropBoxRotation(pcl::CropBox<Point>& box,
                               float roll, float pitch, float yaw) {
    box.setRotation(Eigen::Vector3f(roll, pitch, yaw));
}

}  // namespace pclcompat
