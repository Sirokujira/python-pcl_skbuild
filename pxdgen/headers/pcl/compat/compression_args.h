// Mirror header for src/pcl/compat/compression_args.h: the two stream
// entry points of OctreePointCloudCompression as bytes in / bytes out.
#pragma once

#include <memory>
#include <string>

#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

template <typename PointT>
std::string compressCloud(
    pcl::io::OctreePointCloudCompression<PointT>& codec,
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud);

template <typename PointT>
void decompressCloud(pcl::io::OctreePointCloudCompression<PointT>& codec,
                     const std::string& data,
                     pcl::PointCloud<PointT>& out);

// Whether this PCL has OrganizedPointCloudCompression (needs libpng).
bool haveOrganizedCompression();

// OrganizedPointCloudCompression, PNG level -1 (libpng's default) or 0..9;
// throw std::runtime_error where PCL has none.
std::string compressOrganized(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    bool colour, bool mono, int png_level);
std::string compressOrganized(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    bool colour, bool mono, int png_level);
bool decompressOrganized(const std::string& data,
                         pcl::PointCloud<pcl::PointXYZ>& out);
bool decompressOrganized(const std::string& data,
                         pcl::PointCloud<pcl::PointXYZRGBA>& out);

}  // namespace pclcompat
