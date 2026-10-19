// Mirror header: self-contained stand-in for <pcl/io/obj_io.h>, the
// point-cloud reader only. OBJ WRITING takes a PolygonMesh in PCL (see
// compat/mesh_args.h for meshes), so clouds are read-only in this format.
#pragma once

#include <string>

#include <pcl/point_cloud.h>

namespace pcl {
namespace io {

template <typename PointT>
int loadOBJFile(const std::string& file_name, PointCloud<PointT>& cloud);

}  // namespace io
}  // namespace pcl
