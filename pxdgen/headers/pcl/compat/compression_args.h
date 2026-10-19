// Mirror header for src/pcl/compat/compression_args.h: the two stream
// entry points of OctreePointCloudCompression as bytes in / bytes out.
#pragma once

#include <memory>
#include <string>

#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/point_cloud.h>

namespace pclcompat {

template <typename PointT>
std::string compressCloud(
    pcl::io::OctreePointCloudCompression<PointT>& codec,
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud);

template <typename PointT>
void decompressCloud(pcl::io::OctreePointCloudCompression<PointT>& codec,
                     const std::string& data,
                     pcl::PointCloud<PointT>& out);

}  // namespace pclcompat
