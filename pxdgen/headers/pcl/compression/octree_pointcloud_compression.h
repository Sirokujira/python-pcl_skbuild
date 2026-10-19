// Mirror header: self-contained stand-in for
// <pcl/compression/octree_pointcloud_compression.h>.
//
// Octree-based point cloud compression for transmission or storage. Its
// encode/decode entry points take std::ostream / std::istream, which
// Cython cannot hand over, so they live in compat/compression_args.h
// (bytes in, bytes out); only the constructor is declared here. The
// constructor's defaults are the wrapper's defaults.
#pragma once

#include <pcl/compression/compression_profiles.h>

namespace pcl {
namespace io {

template <typename PointT>
class OctreePointCloudCompression {
public:
    OctreePointCloudCompression(compression_Profiles_e compressionProfile_arg,
                                bool showStatistics_arg,
                                double pointResolution_arg,
                                double octreeResolution_arg,
                                bool doVoxelGridDownDownSampling_arg,
                                unsigned int iFrameRate_arg,
                                bool doColorEncoding_arg,
                                unsigned char colorBitResolution_arg);
};

}  // namespace io
}  // namespace pcl
