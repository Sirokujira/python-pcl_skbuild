// C++ shim: OctreePointCloudCompression encodes to a std::ostream and
// decodes from a std::istream, neither of which Cython can hand over.
// Both are hidden behind a std::string (bytes on the Python side); the
// codec object itself is what the pxd declares.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>

#include <pcl/compression/octree_pointcloud_compression.h>
// organized_pointcloud_compression.h is installed only when PCL found
// libpng (Homebrew's PCL has none), so its presence is the test.
#if defined(__has_include)
#if __has_include(<pcl/compression/organized_pointcloud_compression.h>)
#include <pcl/compression/organized_pointcloud_compression.h>
#define PCLCOMPAT_HAVE_ORGANIZED_COMPRESSION 1
#endif
#endif
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

/// True when this PCL ships OrganizedPointCloudCompression (it needs
/// libpng at PCL's build time).
inline bool haveOrganizedCompression() {
#ifdef PCLCOMPAT_HAVE_ORGANIZED_COMPRESSION
    return true;
#else
    return false;
#endif
}

#ifdef PCLCOMPAT_HAVE_ORGANIZED_COMPRESSION
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

/// The decoder rebuilds the cloud with push_back after setting its width
/// and height, and push_back (PCL 1.11+) resets a cloud to height 1, so
/// every decoded cloud comes back unorganized (same on master). The
/// original size sits right after the frame identifier -- two uint32s --
/// and is put back here.
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
    const std::string id = "<PCL-ORG-COMPRESSED>";
    const std::size_t at = data.find(id);
    if (at != std::string::npos && data.size() >= at + id.size() + 8) {
        std::uint32_t width = 0, height = 0;
        std::memcpy(&width, data.data() + at + id.size(), 4);
        std::memcpy(&height, data.data() + at + id.size() + 4, 4);
        if (static_cast<std::size_t>(width) * height == out.size()) {
            out.width = width;
            out.height = height;
        }
    }
    return true;
}

#else
template <typename PointT>
inline std::string compressOrganized(
    const std::shared_ptr<pcl::PointCloud<PointT>>&, bool, bool, int) {
    throw std::runtime_error(
        "this PCL was built without libpng: no OrganizedPointCloudCompression");
}

template <typename PointT>
inline bool decompressOrganized(const std::string&, pcl::PointCloud<PointT>&) {
    throw std::runtime_error(
        "this PCL was built without libpng: no OrganizedPointCloudCompression");
}
#endif

}  // namespace pclcompat
