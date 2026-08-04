// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/eigen_args.h.
#pragma once

#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/sac_segmentation_normals.h>

namespace pclcompat {

void setCropBoxMin(pcl::CropBox<pcl::PointXYZ>& box,
                   float x, float y, float z);
void setCropBoxMax(pcl::CropBox<pcl::PointXYZ>& box,
                   float x, float y, float z);
void setCropBoxTranslation(pcl::CropBox<pcl::PointXYZ>& box,
                           float x, float y, float z);
void setCropBoxRotation(pcl::CropBox<pcl::PointXYZ>& box,
                        float roll, float pitch, float yaw);

void setSegmentationAxis(pcl::SACSegmentation<pcl::PointXYZ>& seg,
                         float x, float y, float z);
void setSegmentationAxis(
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal>& seg,
    float x, float y, float z);

}  // namespace pclcompat
