// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/eigen_args.h.
#pragma once

#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>

namespace pclcompat {

void setCropBoxMin(pcl::CropBox<pcl::PointXYZ>& box,
                   float x, float y, float z);
void setCropBoxMax(pcl::CropBox<pcl::PointXYZ>& box,
                   float x, float y, float z);
void setCropBoxTranslation(pcl::CropBox<pcl::PointXYZ>& box,
                           float x, float y, float z);
void setCropBoxRotation(pcl::CropBox<pcl::PointXYZ>& box,
                        float roll, float pitch, float yaw);

}  // namespace pclcompat
