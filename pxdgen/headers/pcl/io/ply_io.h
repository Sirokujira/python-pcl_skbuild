// Mirror header: self-contained stand-in for <pcl/io/ply_io.h>.
//
// Only the PointCloud<PointT> overloads are declared; the PCLPointCloud2
// and PolygonMesh ones need types this package does not wrap yet.
#pragma once

#include <string>

#include <pcl/point_cloud.h>

namespace pcl {
namespace io {

template <typename PointT>
int loadPLYFile(const std::string &file_name, PointCloud<PointT> &cloud);

template <typename PointT>
int savePLYFile(const std::string &file_name,
                const PointCloud<PointT> &cloud,
                bool binary_mode = false);

template <typename PointT>
int savePLYFileASCII(const std::string &file_name,
                     const PointCloud<PointT> &cloud);

template <typename PointT>
int savePLYFileBinary(const std::string &file_name,
                      const PointCloud<PointT> &cloud);

}  // namespace io
}  // namespace pcl
