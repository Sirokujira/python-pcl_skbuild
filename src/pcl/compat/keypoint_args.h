// Reach HarrisKeypoint3D's nested ResponseMethod enum from Cython.
//
// `setMethod` takes `HarrisKeypoint3D<PointInT, PointOutT>::ResponseMethod`
// — an enum nested inside a class TEMPLATE. Cython has no syntax for
// naming that (`Klass[A, B].MEMBER` is rejected), and C++ will not
// implicitly convert an int to it, so neither side can bridge the gap
// alone.
//
// The shim takes an int and does the cast, and re-exports the enumerators
// as namespace-scope constants so their values come from the header
// rather than being copied into Python and left to drift.
#pragma once

#include <memory>

#include <pcl/keypoints/harris_2d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/smoothed_surfaces_keypoint.h>
#include <pcl/keypoints/trajkovic_2d.h>
#include <pcl/keypoints/trajkovic_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace pclcompat {

using HarrisXYZI = pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;

const int HARRIS_METHOD_HARRIS = HarrisXYZI::HARRIS;
const int HARRIS_METHOD_NOBLE = HarrisXYZI::NOBLE;
const int HARRIS_METHOD_LOWE = HarrisXYZI::LOWE;
const int HARRIS_METHOD_TOMASI = HarrisXYZI::TOMASI;
const int HARRIS_METHOD_CURVATURE = HarrisXYZI::CURVATURE;

inline void setHarrisMethod(HarrisXYZI& detector, int method) {
    detector.setMethod(static_cast<HarrisXYZI::ResponseMethod>(method));
}

// The same for the 2-D Harris detector (its ResponseMethod has no
// CURVATURE) and for Trajkovic's ComputationMethod.
using Harris2DRGBA = pcl::HarrisKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>;
using Trajkovic2DRGBA = pcl::TrajkovicKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>;
using Trajkovic3DXYZ = pcl::TrajkovicKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;

inline void setHarris2DMethod(Harris2DRGBA& detector, int method) {
    detector.setMethod(static_cast<Harris2DRGBA::ResponseMethod>(method));
}

const int TRAJKOVIC_FOUR_CORNERS = Trajkovic2DRGBA::FOUR_CORNERS;
const int TRAJKOVIC_EIGHT_CORNERS = Trajkovic2DRGBA::EIGHT_CORNERS;

inline void setTrajkovic2DMethod(Trajkovic2DRGBA& detector, int method) {
    detector.setMethod(static_cast<Trajkovic2DRGBA::ComputationMethod>(method));
}

inline void setTrajkovic3DMethod(Trajkovic3DXYZ& detector, int method) {
    detector.setMethod(static_cast<Trajkovic3DXYZ::ComputationMethod>(method));
}

// addSmoothedPointCloud takes the tree and the scale by non-const
// reference; the tree is one more object nothing above needs to see.
inline void addSmoothedCloud(
    pcl::SmoothedSurfacesKeypoint<pcl::PointXYZ, pcl::Normal>& detector,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    float scale) {
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);
    detector.addSmoothedPointCloud(cloud, normals, tree, scale);
}

}  // namespace pclcompat
