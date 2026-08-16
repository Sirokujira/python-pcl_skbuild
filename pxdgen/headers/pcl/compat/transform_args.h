// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/transform_args.h (reached as
// "pcl/compat/transform_args.h" because CMake adds src/ to the
// includes).
//
// The real header names Eigen::Matrix4f throughout; this one takes the
// matrix as 16 floats, so pxd generation still runs on a machine with
// neither PCL nor Eigen installed.
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

void transformCloud(const pcl::PointCloud<pcl::PointXYZ>& in,
                    pcl::PointCloud<pcl::PointXYZ>& out,
                    const float* matrix16);
void transformCloud(const pcl::PointCloud<pcl::PointXYZI>& in,
                    pcl::PointCloud<pcl::PointXYZI>& out,
                    const float* matrix16);
void transformCloud(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                    pcl::PointCloud<pcl::PointXYZRGB>& out,
                    const float* matrix16);
void transformCloud(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                    pcl::PointCloud<pcl::PointXYZRGBA>& out,
                    const float* matrix16);

bool isRigidTransform(const float* matrix16, float tolerance);

}  // namespace pclcompat
