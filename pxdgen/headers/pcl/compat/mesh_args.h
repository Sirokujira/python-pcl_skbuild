// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/mesh_args.h (reached as "pcl/compat/mesh_args.h"
// because CMake adds src/ to the includes).
//
// The real header converts through pcl::PolygonMesh, whose points are a
// PCLPointCloud2 blob and whose faces are vectors of pcl::Vertices;
// this one names only the two entry points and the format constants, so
// pxd generation still runs without PCL installed.
#pragma once

#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// Values come from the real header at C++ compile time; the pxd only
// needs the names.
const int MESH_FORMAT_PLY = 0;
const int MESH_FORMAT_OBJ = 1;
const int MESH_FORMAT_VTK = 2;

int saveMesh(const std::string& path,
             const pcl::PointCloud<pcl::PointXYZ>& cloud,
             const std::vector<int>& indices,
             const std::vector<int>& counts,
             int format, bool binary);

int loadMesh(const std::string& path, int format,
             pcl::PointCloud<pcl::PointXYZ>& out_cloud,
             std::vector<int>& out_indices,
             std::vector<int>& out_counts);

}  // namespace pclcompat
