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

#include <pcl/keypoints/agast_2d.h>
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

// AgastKeypoint2D's detector types.
const int AGAST_7_12S = 0;
const int AGAST_5_8 = 1;
const int OAST_9_16 = 2;

/// AgastKeypoint2D::detectKeypoints builds its detector once -- from the
/// first image's size, threshold and maximum value -- and keeps it. A
/// later image of another size is then read with the first one's row
/// length (a 120-wide frame after a 40-wide one: 0 corners where a fresh
/// detector finds 4; a smaller one reads past the end), and
/// setThreshold after the first compute does nothing. agastDetect gives
/// it a fresh detector of `kind` for every call.
inline void agastDetect(pcl::AgastKeypoint2D<pcl::PointXYZRGBA, pcl::PointUV>& detector,
                        int kind, pcl::PointCloud<pcl::PointUV>& out) {
    namespace agast = pcl::keypoints::agast;
    const auto input = detector.getInputCloud();
    const std::size_t width = input->width;
    const std::size_t height = input->height;
    const double threshold = detector.getThreshold();
    const double bmax = detector.getMaxDataValue();
    agast::AbstractAgastDetector::Ptr fresh;
    if (kind == AGAST_5_8) {
        fresh.reset(new agast::AgastDetector5_8(width, height, threshold, bmax));
    } else if (kind == OAST_9_16) {
        fresh.reset(new agast::OastDetector9_16(width, height, threshold, bmax));
    } else {
        fresh.reset(new agast::AgastDetector7_12s(width, height, threshold, bmax));
    }
    detector.setAgastDetector(fresh);
    detector.compute(out);
}

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
