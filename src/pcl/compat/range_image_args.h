// Build a RangeImage and run NARF keypoint detection from Cython.
//
// This chain is the densest concentration of things Cython cannot say in
// all of PCL:
//
//  * `createFromPointCloud` takes an `Eigen::Affine3f` sensor pose and a
//    `CoordinateFrame` enum nested inside RangeImage.
//  * `RangeImage` derives from `PointCloud<PointWithRange>` and is read
//    through operator() on a pixel grid.
//  * `NarfKeypoint` needs a `RangeImageBorderExtractor` wired to the same
//    image, its parameters live in a nested `Parameters` struct, and its
//    output is a `PointCloud<int>`.
//
// Rather than mirror four types to run one algorithm, the shim owns the
// whole pipeline and hands back what a caller actually wants: the range
// image as a flat float buffer, and the keypoints as indices.
#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/bearing_angle_image.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/features/narf_descriptor.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using RangeImagePtr = std::shared_ptr<pcl::RangeImage>;

const int RANGE_IMAGE_CAMERA_FRAME = pcl::RangeImage::CAMERA_FRAME;
const int RANGE_IMAGE_LASER_FRAME = pcl::RangeImage::LASER_FRAME;

/// Angles are radians. The sensor sits at the origin looking along the
/// coordinate frame's axis, which is what every PCL example does and the
/// only pose an Eigen-free signature can offer.
inline RangeImagePtr makeRangeImage(const Cloud& cloud,
                                    float angular_resolution,
                                    float max_angle_width,
                                    float max_angle_height,
                                    int coordinate_frame,
                                    float noise_level,
                                    float min_range,
                                    int border_size) {
    RangeImagePtr image(new pcl::RangeImage());
    image->createFromPointCloud(
        cloud, angular_resolution, max_angle_width, max_angle_height,
        Eigen::Affine3f::Identity(),
        static_cast<pcl::RangeImage::CoordinateFrame>(coordinate_frame),
        noise_level, min_range, border_size);
    return image;
}

inline void rangeImageSetUnseenToMaxRange(pcl::RangeImage& image) {
    image.setUnseenToMaxRange();
}

inline int rangeImageWidth(const pcl::RangeImage& image) {
    return static_cast<int>(image.width);
}

inline int rangeImageHeight(const pcl::RangeImage& image) {
    return static_cast<int>(image.height);
}

/// Ranges row-major, one per pixel; unobserved pixels come back as
/// infinity, which is how PCL marks them.
inline void rangeImageRanges(const pcl::RangeImage& image,
                             std::vector<float>& out) {
    out.resize(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
        out[i] = image[i].range;
    }
}

/// The 3-D points behind those pixels, in the same order.
inline void rangeImagePoints(const pcl::RangeImage& image,
                             std::vector<Point>& out) {
    out.resize(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
        out[i].x = image[i].x;
        out[i].y = image[i].y;
        out[i].z = image[i].z;
    }
}

/// Run border extraction and NARF over *image*; fills the keypoint
/// indices into the range image and returns how many were found.
inline int narfKeypoints(pcl::RangeImage& image, float support_size,
                         bool add_points_on_straight_edges,
                         std::vector<int>& out) {
    pcl::RangeImageBorderExtractor border_extractor;
    pcl::NarfKeypoint detector(&border_extractor);
    detector.setRangeImage(&image);
    detector.getParameters().support_size = support_size;
    detector.getParameters().add_points_on_straight_edges =
        add_points_on_straight_edges;

    pcl::PointCloud<int> keypoints;
    detector.compute(keypoints);
    out.assign(keypoints.points.begin(), keypoints.points.end());
    return static_cast<int>(out.size());
}

// --- RangeImagePlanar / RangeImageSpherical -----------------------------
//
// Both are RangeImage subclasses that only change how a pixel maps to a
// direction, so they travel up as a RangeImage pointer and every
// RangeImage helper above works on them unchanged.

inline RangeImagePtr makeRangeImagePlanar(const Cloud& cloud, int width,
                                          int height, float center_x,
                                          float center_y,
                                          float focal_length_x,
                                          float focal_length_y,
                                          int coordinate_frame,
                                          float noise_level,
                                          float min_range) {
    std::shared_ptr<pcl::RangeImagePlanar> image(new pcl::RangeImagePlanar());
    image->createFromPointCloudWithFixedSize(
        cloud, width, height, center_x, center_y, focal_length_x,
        focal_length_y, Eigen::Affine3f::Identity(),
        static_cast<pcl::RangeImage::CoordinateFrame>(coordinate_frame),
        noise_level, min_range);
    return image;
}

inline RangeImagePtr makeRangeImagePlanarFromDepth(const float* depth,
                                                   int width, int height,
                                                   float center_x,
                                                   float center_y,
                                                   float focal_length_x,
                                                   float focal_length_y) {
    std::shared_ptr<pcl::RangeImagePlanar> image(new pcl::RangeImagePlanar());
    image->setDepthImage(depth, width, height, center_x, center_y,
                         focal_length_x, focal_length_y);
    return image;
}

/// fx, fy, cx, cy of a planar image; false when the image is not planar.
inline bool rangeImagePlanarIntrinsics(const pcl::RangeImage& image,
                                       float* out4) {
    const auto* planar = dynamic_cast<const pcl::RangeImagePlanar*>(&image);
    if (planar == nullptr) {
        return false;
    }
    out4[0] = planar->getFocalLengthX();
    out4[1] = planar->getFocalLengthY();
    out4[2] = planar->getCenterX();
    out4[3] = planar->getCenterY();
    return true;
}

inline RangeImagePtr makeRangeImageSpherical(const Cloud& cloud,
                                             float angular_resolution,
                                             float max_angle_width,
                                             float max_angle_height,
                                             int coordinate_frame,
                                             float noise_level,
                                             float min_range,
                                             int border_size) {
#ifdef _WIN32
    // RangeImageSpherical's constructor is inline, so its vtable is built
    // in the caller's module, and that vtable needs RangeImage's
    // getSubImage / getHalfImage (not exported from pcl_common) and
    // inline overrides that read RangeImage's static lookup tables (data
    // a consumer cannot link: PCL_EXPORTS is empty outside PCL, so there
    // is no dllimport). No module outside PCL can link it on Windows.
    (void)cloud;
    (void)angular_resolution;
    (void)max_angle_width;
    (void)max_angle_height;
    (void)coordinate_frame;
    (void)noise_level;
    (void)min_range;
    (void)border_size;
    throw std::runtime_error(
        "RangeImageSpherical cannot be linked from outside PCL on Windows "
        "(its base's virtuals and lookup tables are not exported)");
#else
    std::shared_ptr<pcl::RangeImageSpherical> image(
        new pcl::RangeImageSpherical());
    image->createFromPointCloud(
        cloud, angular_resolution, max_angle_width, max_angle_height,
        Eigen::Affine3f::Identity(),
        static_cast<pcl::RangeImage::CoordinateFrame>(coordinate_frame),
        noise_level, min_range, border_size);
    return image;
#endif
}

// --- borders and NARF descriptors ----------------------------------------

const int BORDER_TRAIT_OBSTACLE_BORDER = pcl::BORDER_TRAIT__OBSTACLE_BORDER;
const int BORDER_TRAIT_SHADOW_BORDER = pcl::BORDER_TRAIT__SHADOW_BORDER;
const int BORDER_TRAIT_VEIL_POINT = pcl::BORDER_TRAIT__VEIL_POINT;
const int BORDER_TRAIT_SHADOW_BORDER_TOP = pcl::BORDER_TRAIT__SHADOW_BORDER_TOP;
const int BORDER_TRAIT_SHADOW_BORDER_RIGHT = pcl::BORDER_TRAIT__SHADOW_BORDER_RIGHT;
const int BORDER_TRAIT_SHADOW_BORDER_BOTTOM = pcl::BORDER_TRAIT__SHADOW_BORDER_BOTTOM;
const int BORDER_TRAIT_SHADOW_BORDER_LEFT = pcl::BORDER_TRAIT__SHADOW_BORDER_LEFT;
const int BORDER_TRAIT_OBSTACLE_BORDER_TOP = pcl::BORDER_TRAIT__OBSTACLE_BORDER_TOP;
const int BORDER_TRAIT_OBSTACLE_BORDER_RIGHT = pcl::BORDER_TRAIT__OBSTACLE_BORDER_RIGHT;
const int BORDER_TRAIT_OBSTACLE_BORDER_BOTTOM = pcl::BORDER_TRAIT__OBSTACLE_BORDER_BOTTOM;
const int BORDER_TRAIT_OBSTACLE_BORDER_LEFT = pcl::BORDER_TRAIT__OBSTACLE_BORDER_LEFT;
const int BORDER_TRAIT_VEIL_POINT_TOP = pcl::BORDER_TRAIT__VEIL_POINT_TOP;
const int BORDER_TRAIT_VEIL_POINT_RIGHT = pcl::BORDER_TRAIT__VEIL_POINT_RIGHT;
const int BORDER_TRAIT_VEIL_POINT_BOTTOM = pcl::BORDER_TRAIT__VEIL_POINT_BOTTOM;
const int BORDER_TRAIT_VEIL_POINT_LEFT = pcl::BORDER_TRAIT__VEIL_POINT_LEFT;

/// One BorderTraits bitset per pixel, as its integer value: bit
/// BORDER_TRAIT_X is set when the pixel has trait X.
inline void rangeImageBorderTraits(pcl::RangeImage& image,
                                   int pixel_radius_borders,
                                   float minimum_border_probability,
                                   std::vector<unsigned int>& out) {
    pcl::RangeImageBorderExtractor extractor(&image);
    extractor.getParameters().pixel_radius_borders = pixel_radius_borders;
    extractor.getParameters().minimum_border_probability =
        minimum_border_probability;
    pcl::PointCloud<pcl::BorderDescription> borders;
    extractor.compute(borders);
    out.resize(borders.size());
    for (std::size_t i = 0; i < borders.size(); ++i) {
        out[i] = static_cast<unsigned int>(borders[i].traits.to_ulong());
    }
}

/// NARF descriptors at the given pixels: 42 floats each (x, y, z, roll,
/// pitch, yaw, then the 36 descriptor values). With rotation_invariant a
/// pixel can yield one descriptor per dominant orientation, so the count
/// returned can exceed indices.size().
inline int narfDescriptors(pcl::RangeImage& image,
                           const std::vector<int>& indices,
                           float support_size, bool rotation_invariant,
                           std::vector<float>& out) {
    pcl::Indices pixels(indices.begin(), indices.end());
    pcl::NarfDescriptor descriptor(&image, &pixels);
    descriptor.getParameters().support_size = support_size;
    descriptor.getParameters().rotation_invariant = rotation_invariant;
    pcl::PointCloud<pcl::Narf36> features;
    descriptor.compute(features);

    out.resize(features.size() * 42);
    for (std::size_t i = 0; i < features.size(); ++i) {
        const pcl::Narf36& f = features[i];
        float* row = &out[i * 42];
        row[0] = f.x;
        row[1] = f.y;
        row[2] = f.z;
        row[3] = f.roll;
        row[4] = f.pitch;
        row[5] = f.yaw;
        for (int j = 0; j < 36; ++j) {
            row[6 + j] = f.descriptor[j];
        }
    }
    return static_cast<int>(features.size());
}

// --- bearing angle images ---------------------------------------------------

/// pcl::BearingAngleImage of an organized cloud: per pixel, the angle
/// between the laser beam and the surface step to its neighbour, as a
/// grey level (0..255 for 0..180 degrees), one byte per pixel.
inline void bearingAngleImage(pcl::PointCloud<pcl::PointXYZ>& cloud,
                              std::vector<unsigned char>& grey,
                              unsigned int& width, unsigned int& height) {
    pcl::BearingAngleImage image;
    image.generateBAImage(cloud);
    width = image.width;
    height = image.height;
    grey.resize(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
        grey[i] = image[i].r;
    }
}

/// BearingAngleImage::getAngle, in degrees.
inline double bearingAngle(const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
    pcl::BearingAngleImage image;
    return image.getAngle(a, b);
}

}  // namespace pclcompat
