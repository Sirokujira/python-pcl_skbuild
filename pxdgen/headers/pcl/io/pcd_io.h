// Mirror header: self-contained stand-in for <pcl/io/pcd_io.h> — the
// PCD load/save function templates wrapped so far. Only NAMES and TYPES
// must match the real PCL API; the real declarations are used at C++
// compile time (extern_from points at pcl/io/pcd_io.h).
#pragma once

#include <string>

#include <pcl/point_cloud.h>

namespace pcl {
namespace io {

template <typename PointT>
int loadPCDFile(const std::string &file_name, PointCloud<PointT> &cloud);

template <typename PointT>
int savePCDFile(const std::string &file_name,
                const PointCloud<PointT> &cloud,
                bool binary_mode = false);

template <typename PointT>
int savePCDFileASCII(const std::string &file_name,
                     const PointCloud<PointT> &cloud);

template <typename PointT>
int savePCDFileBinary(const std::string &file_name,
                      PointCloud<PointT> &cloud);

}  // namespace io
}  // namespace pcl
