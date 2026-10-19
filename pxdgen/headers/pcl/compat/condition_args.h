// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/condition_args.h.
//
// ConditionalRemoval with the condition tree passed as a flat prefix
// encoding (see the shim for the layout), rebuilt for the cloud's point
// type. One overload per point type the wrapper filters.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int COMPARE_GT = 0;
const int COMPARE_GE = 1;
const int COMPARE_LT = 2;
const int COMPARE_LE = 3;
const int COMPARE_EQ = 4;

const int CONDITION_AND = 0;
const int CONDITION_OR = 1;
const int COMPARISON_FIELD = 2;
const int COMPARISON_TF_QUADRATIC = 3;
const int COMPARISON_PACKED_RGB = 4;
const int COMPARISON_PACKED_HSI = 5;

void conditionalRemoval(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::vector<int>& kinds, const std::vector<int>& ints,
    const std::vector<double>& numbers, const std::vector<std::string>& names,
    bool keep_organized, float user_value,
    pcl::PointCloud<pcl::PointXYZ>& out);

void conditionalRemoval(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>& cloud,
    const std::vector<int>& kinds, const std::vector<int>& ints,
    const std::vector<double>& numbers, const std::vector<std::string>& names,
    bool keep_organized, float user_value,
    pcl::PointCloud<pcl::PointXYZRGB>& out);

void conditionalRemoval(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    const std::vector<int>& kinds, const std::vector<int>& ints,
    const std::vector<double>& numbers, const std::vector<std::string>& names,
    bool keep_organized, float user_value,
    pcl::PointCloud<pcl::PointXYZRGBA>& out);

}  // namespace pclcompat
