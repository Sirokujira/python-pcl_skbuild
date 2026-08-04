// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/range_image_args.h.
//
// RangeImage appears only as an opaque handle: nothing above this layer
// calls a method on one, so it needs no mirror of its own.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

namespace pclcompat {

const int RANGE_IMAGE_CAMERA_FRAME = 0;
const int RANGE_IMAGE_LASER_FRAME = 1;

std::shared_ptr<pcl::RangeImage> makeRangeImage(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    float angular_resolution, float max_angle_width, float max_angle_height,
    int coordinate_frame, float noise_level, float min_range,
    int border_size);

void rangeImageSetUnseenToMaxRange(pcl::RangeImage& image);
int rangeImageWidth(const pcl::RangeImage& image);
int rangeImageHeight(const pcl::RangeImage& image);
void rangeImageRanges(const pcl::RangeImage& image, std::vector<float>& out);
void rangeImagePoints(const pcl::RangeImage& image,
                      std::vector<pcl::PointXYZ>& out);

int narfKeypoints(pcl::RangeImage& image, float support_size,
                  bool add_points_on_straight_edges, std::vector<int>& out);

}  // namespace pclcompat
