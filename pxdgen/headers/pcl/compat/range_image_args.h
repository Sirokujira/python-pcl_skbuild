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

// RangeImagePlanar / RangeImageSpherical, returned as the base class.
std::shared_ptr<pcl::RangeImage> makeRangeImagePlanar(
    const pcl::PointCloud<pcl::PointXYZ>& cloud, int width, int height,
    float center_x, float center_y, float focal_length_x,
    float focal_length_y, int coordinate_frame, float noise_level,
    float min_range);
std::shared_ptr<pcl::RangeImage> makeRangeImagePlanarFromDepth(
    const float* depth, int width, int height, float center_x,
    float center_y, float focal_length_x, float focal_length_y);
bool rangeImagePlanarIntrinsics(const pcl::RangeImage& image, float* out4);
std::shared_ptr<pcl::RangeImage> makeRangeImageSpherical(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    float angular_resolution, float max_angle_width, float max_angle_height,
    int coordinate_frame, float noise_level, float min_range,
    int border_size);

// pcl::BorderTrait bit positions.
const int BORDER_TRAIT_OBSTACLE_BORDER = 0;
const int BORDER_TRAIT_SHADOW_BORDER = 1;
const int BORDER_TRAIT_VEIL_POINT = 2;
const int BORDER_TRAIT_SHADOW_BORDER_TOP = 3;
const int BORDER_TRAIT_SHADOW_BORDER_RIGHT = 4;
const int BORDER_TRAIT_SHADOW_BORDER_BOTTOM = 5;
const int BORDER_TRAIT_SHADOW_BORDER_LEFT = 6;
const int BORDER_TRAIT_OBSTACLE_BORDER_TOP = 7;
const int BORDER_TRAIT_OBSTACLE_BORDER_RIGHT = 8;
const int BORDER_TRAIT_OBSTACLE_BORDER_BOTTOM = 9;
const int BORDER_TRAIT_OBSTACLE_BORDER_LEFT = 10;
const int BORDER_TRAIT_VEIL_POINT_TOP = 11;
const int BORDER_TRAIT_VEIL_POINT_RIGHT = 12;
const int BORDER_TRAIT_VEIL_POINT_BOTTOM = 13;
const int BORDER_TRAIT_VEIL_POINT_LEFT = 14;

void rangeImageBorderTraits(pcl::RangeImage& image, int pixel_radius_borders,
                            float minimum_border_probability,
                            std::vector<unsigned int>& out);

int narfDescriptors(pcl::RangeImage& image, const std::vector<int>& indices,
                    float support_size, bool rotation_invariant,
                    std::vector<float>& out);

}  // namespace pclcompat
