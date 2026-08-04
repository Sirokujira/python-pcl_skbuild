// Mirror header: self-contained stand-in for
// <pcl/range_image/range_image.h>.
//
// Opaque on purpose. Creating one needs an Eigen::Affine3f pose and a
// nested CoordinateFrame enum, and reading one means indexing a pixel
// grid of PointWithRange — all of it goes through
// pcl/compat/range_image_args.h. This exists so the shim's signatures
// have a type to name.
#pragma once

namespace pcl {

class RangeImage {};

}  // namespace pcl
