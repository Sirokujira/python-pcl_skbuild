// C++ shim: OctreePointCloudCompression encodes to a std::ostream and
// decodes from a std::istream, neither of which Cython can hand over.
// Both are hidden behind a std::string (bytes on the Python side); the
// codec object itself is what the pxd declares.
#pragma once

#include <memory>
#include <sstream>
#include <string>

#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/compression/organized_pointcloud_compression.h>
#include <pcl/point_cloud.h>

namespace pclcompat {

template <typename PointT>
inline std::string compressCloud(
    pcl::io::OctreePointCloudCompression<PointT>& codec,
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud) {
    std::stringstream buffer;
    codec.encodePointCloud(cloud, buffer);
    return buffer.str();
}

template <typename PointT>
inline void decompressCloud(pcl::io::OctreePointCloudCompression<PointT>& codec,
                            const std::string& data,
                            pcl::PointCloud<PointT>& out) {
    std::stringstream buffer(data);
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    codec.decodePointCloud(buffer, cloud);
    out = *cloud;
}

/// OrganizedPointCloudCompression: the depth image (and optionally the
/// colour, as RGB or grey) PNG-compressed at `png_level` (-1 libpng's
/// default, 0..9). The cloud must be organized.
template <typename PointT>
inline std::string compressOrganized(
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud, bool colour,
    bool mono, int png_level) {
    pcl::io::OrganizedPointCloudCompression<PointT> codec;
    std::stringstream buffer;
    codec.encodePointCloud(cloud, buffer, colour, mono, false, png_level);
    return buffer.str();
}

template <typename PointT>
inline bool decompressOrganized(const std::string& data,
                                pcl::PointCloud<PointT>& out) {
    pcl::io::OrganizedPointCloudCompression<PointT> codec;
    std::stringstream buffer(data);
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (!codec.decodePointCloud(buffer, cloud, false)) {
        return false;
    }
    out = *cloud;
    return true;
}

}  // namespace pclcompat
