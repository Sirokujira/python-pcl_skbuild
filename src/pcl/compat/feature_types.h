// C++ shim: the plain names behind pcl::Histogram<N> (an integer
// template argument Cython cannot pass), SUSANKeypoint's intensity
// accessor template argument, and CRHEstimation::setCentroid (an Eigen
// vector). See pxdgen/headers/pcl/compat/feature_types.h for the mirror.
#pragma once

#include <list>
#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/common/norms.h>
#include <pcl/features/crh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/from_meshes.h>
#include <pcl/features/multiscale_feature_persistence.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/pfh_tools.h>
#include <pcl/features/statistical_multiscale_interest_region_extraction.h>
#include <pcl/keypoints/susan.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Histogram153 = pcl::Histogram<153>;
using Histogram90 = pcl::Histogram<90>;
using Histogram20 = pcl::Histogram<20>;
using Histogram32 = pcl::Histogram<32>;
using Histogram135 = pcl::Histogram<135>;

using EdgeBase = pcl::OrganizedEdgeBase<pcl::PointXYZ, pcl::Label>;
const int EDGELABEL_NAN_BOUNDARY = EdgeBase::EDGELABEL_NAN_BOUNDARY;
const int EDGELABEL_OCCLUDING = EdgeBase::EDGELABEL_OCCLUDING;
const int EDGELABEL_OCCLUDED = EdgeBase::EDGELABEL_OCCLUDED;
const int EDGELABEL_HIGH_CURVATURE = EdgeBase::EDGELABEL_HIGH_CURVATURE;
const int EDGELABEL_RGB_CANNY = EdgeBase::EDGELABEL_RGB_CANNY;

const int NORM_L1 = pcl::L1;
const int NORM_L2_SQR = pcl::L2_SQR;
const int NORM_L2 = pcl::L2;
const int NORM_LINF = pcl::LINF;
const int NORM_JM = pcl::JM;
const int NORM_B = pcl::B;
const int NORM_SUBLINEAR = pcl::SUBLINEAR;
const int NORM_CS = pcl::CS;
const int NORM_DIV = pcl::DIV;
const int NORM_PF = pcl::PF;
const int NORM_K = pcl::K;
const int NORM_KL = pcl::KL;
const int NORM_HIK = pcl::HIK;
using SUSANKeypointRGB = pcl::SUSANKeypoint<pcl::PointXYZRGB, pcl::PointXYZ>;

inline void setCRHCentroid(
    pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, Histogram90>& estimator,
    float x, float y, float z) {
    Eigen::Vector4f centroid(x, y, z, 0.0f);
    estimator.setCentroid(centroid);
}

inline void approximateNormals(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                               const std::vector<pcl::Vertices>& polygons,
                               pcl::PointCloud<pcl::Normal>& normals) {
    pcl::features::computeApproximateNormals(cloud, polygons, normals);
}

inline bool pairFeatures(const float* p1, const float* n1, const float* p2,
                         const float* n2, float* out4) {
    const Eigen::Vector4f a(p1[0], p1[1], p1[2], 0.0f);
    const Eigen::Vector4f na(n1[0], n1[1], n1[2], 0.0f);
    const Eigen::Vector4f b(p2[0], p2[1], p2[2], 0.0f);
    const Eigen::Vector4f nb(n2[0], n2[1], n2[2], 0.0f);
    return pcl::computePairFeatures(a, na, b, nb, out4[0], out4[1], out4[2],
                                    out4[3]);
}

inline void multiscaleInterestRegions(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::vector<float>& scales,
    std::vector<std::vector<int>>& out_regions) {
    pcl::StatisticalMultiscaleInterestRegionExtraction<pcl::PointXYZ> extractor;
    extractor.setInputCloud(cloud);
    std::vector<float> scale_values(scales);
    extractor.setScalesVector(scale_values);
    std::list<pcl::IndicesPtr> rois;
    extractor.computeRegionsOfInterest(rois);
    out_regions.clear();
    for (const auto& roi : rois) {
        out_regions.emplace_back(roi->begin(), roi->end());
    }
}

inline void persistentFPFH(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    const std::vector<float>& scales, float alpha, int norm_type,
    pcl::PointCloud<pcl::FPFHSignature33>& out_features,
    std::vector<int>& out_indices) {
    auto fpfh = std::make_shared<
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>>();
    fpfh->setInputCloud(cloud);
    fpfh->setInputNormals(normals);

    pcl::MultiscaleFeaturePersistence<pcl::PointXYZ, pcl::FPFHSignature33> persistence;
    persistence.setInputCloud(cloud);
    persistence.setFeatureEstimator(fpfh);
    std::vector<float> scale_values(scales);
    persistence.setScalesVector(scale_values);
    persistence.setAlpha(alpha);
    persistence.setDistanceMetric(static_cast<pcl::NormType>(norm_type));

    pcl::IndicesPtr indices(new pcl::Indices());
    persistence.determinePersistentFeatures(out_features, indices);
    out_indices.assign(indices->begin(), indices->end());
}

}  // namespace pclcompat
