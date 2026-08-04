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
#include <vector>

#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

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

}  // namespace pclcompat
