// C++ shim: OctreePointCloudCompression encodes to a std::ostream and
// decodes from a std::istream, neither of which Cython can hand over.
// Both are hidden behind a std::string (bytes on the Python side); the
// codec object itself is what the pxd declares.
#pragma once

#include <memory>
#include <sstream>
#include <string>

#include <pcl/compression/octree_pointcloud_compression.h>
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

}  // namespace pclcompat
