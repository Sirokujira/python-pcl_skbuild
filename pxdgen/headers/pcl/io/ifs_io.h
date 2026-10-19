// Mirror header: self-contained stand-in for <pcl/io/ifs_io.h>.
//
// The Indexed Face Set format (Brown's mesh archive). The PointT
// template is declared for the one type the wrapper reads; writing goes
// through pclcompat::saveIFSCloud (PCL's writer leaks PointXYZ padding)
// and meshes through compat/mesh_args.h. 0 on success.
#pragma once

#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {
namespace io {

int loadIFSFile(const std::string& file_name, PointCloud<PointXYZ>& cloud);

}  // namespace io
}  // namespace pcl
