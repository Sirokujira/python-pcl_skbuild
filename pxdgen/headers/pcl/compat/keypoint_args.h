// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/keypoint_args.h.
#pragma once

#include <memory>

#include <pcl/keypoints/harris_2d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/smoothed_surfaces_keypoint.h>
#include <pcl/keypoints/trajkovic_2d.h>
#include <pcl/keypoints/trajkovic_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int HARRIS_METHOD_HARRIS = 1;
const int HARRIS_METHOD_NOBLE = 2;
const int HARRIS_METHOD_LOWE = 3;
const int HARRIS_METHOD_TOMASI = 4;
const int HARRIS_METHOD_CURVATURE = 5;

void setHarrisMethod(
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>& detector,
    int method);

void setHarris2DMethod(
    pcl::HarrisKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>& detector,
    int method);

// TrajkovicKeypoint2D / 3D ComputationMethod.
const int TRAJKOVIC_FOUR_CORNERS = 0;
const int TRAJKOVIC_EIGHT_CORNERS = 1;

void setTrajkovic2DMethod(
    pcl::TrajkovicKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>& detector,
    int method);
void setTrajkovic3DMethod(
    pcl::TrajkovicKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>& detector,
    int method);

// SmoothedSurfacesKeypoint::addSmoothedPointCloud, with the search tree
// it also wants built here.
void addSmoothedCloud(
    pcl::SmoothedSurfacesKeypoint<pcl::PointXYZ, pcl::Normal>& detector,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    float scale);

}  // namespace pclcompat
