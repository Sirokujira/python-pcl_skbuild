// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/texture_args.h (reached as "pcl/compat/texture_args.h"
// because CMake adds src/ to the includes).
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

void textureWithCameras(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                        const std::vector<int>& triangles,
                        const std::vector<double>& cameras,
                        std::vector<int>& group_sizes, std::vector<int>& faces,
                        std::vector<float>& uv);

}  // namespace pclcompat
