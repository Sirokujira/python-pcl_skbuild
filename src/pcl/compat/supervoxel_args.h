// C++ shim: SupervoxelClustering::extract fills a std::map of Supervoxel
// objects and getSupervoxelAdjacency a std::multimap; LCCPSegmentation
// consumes both. None of that has a pxd spelling, so the results stay
// here in a SupervoxelSet and come out as flat vectors. See
// pxdgen/headers/pcl/compat/supervoxel_args.h for the mirror.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>

namespace pclcompat {

template <typename PointT>
struct SupervoxelSet {
    std::map<std::uint32_t, typename pcl::Supervoxel<PointT>::Ptr> clusters;
    std::multimap<std::uint32_t, std::uint32_t> adjacency;
};

template <typename PointT>
inline void supervoxelExtract(pcl::SupervoxelClustering<PointT>& clustering,
                              SupervoxelSet<PointT>& out) {
    out.clusters.clear();
    out.adjacency.clear();
    clustering.extract(out.clusters);
    clustering.getSupervoxelAdjacency(out.adjacency);
}

template <typename PointT>
inline void supervoxelSummary(const SupervoxelSet<PointT>& set,
                              std::vector<unsigned int>& labels,
                              std::vector<float>& centroids,
                              std::vector<float>& normals,
                              std::vector<unsigned int>& adjacency) {
    labels.clear();
    centroids.clear();
    normals.clear();
    adjacency.clear();
    for (const auto& item : set.clusters) {
        const pcl::Supervoxel<PointT>& sv = *item.second;
        labels.push_back(item.first);
        centroids.push_back(sv.centroid_.x);
        centroids.push_back(sv.centroid_.y);
        centroids.push_back(sv.centroid_.z);
        normals.push_back(sv.normal_.normal_x);
        normals.push_back(sv.normal_.normal_y);
        normals.push_back(sv.normal_.normal_z);
        normals.push_back(sv.normal_.curvature);
    }
    for (const auto& edge : set.adjacency) {
        adjacency.push_back(edge.first);
        adjacency.push_back(edge.second);
    }
}

template <typename PointT>
inline void supervoxelPointLabels(
    const pcl::SupervoxelClustering<PointT>& clustering,
    std::vector<unsigned int>& out) {
    typename pcl::PointCloud<pcl::PointXYZL>::Ptr labeled =
        clustering.getLabeledCloud();
    out.clear();
    out.reserve(labeled->size());
    for (const auto& point : *labeled) {
        out.push_back(point.label);
    }
}

template <typename PointT>
inline void lccpSegment(const SupervoxelSet<PointT>& set,
                        const pcl::SupervoxelClustering<PointT>& clustering,
                        float concavity_tolerance, bool smoothness_check,
                        float voxel_resolution, float seed_resolution,
                        float smoothness_threshold, bool sanity_check,
                        unsigned int k_factor, unsigned int min_segment_size,
                        std::vector<unsigned int>& point_labels,
                        std::vector<unsigned int>& supervoxel_to_segment) {
    pcl::LCCPSegmentation<PointT> lccp;
    lccp.setConcavityToleranceThreshold(concavity_tolerance);
    lccp.setSmoothnessCheck(smoothness_check, voxel_resolution,
                            seed_resolution, smoothness_threshold);
    lccp.setSanityCheck(sanity_check);
    lccp.setKFactor(k_factor);
    lccp.setMinSegmentSize(min_segment_size);
    lccp.setInputSupervoxels(set.clusters, set.adjacency);
    lccp.segment();

    typename pcl::PointCloud<pcl::PointXYZL>::Ptr labeled =
        clustering.getLabeledCloud();
    lccp.relabelCloud(*labeled);
    point_labels.clear();
    point_labels.reserve(labeled->size());
    for (const auto& point : *labeled) {
        point_labels.push_back(point.label);
    }

    std::map<std::uint32_t, std::uint32_t> mapping;
    lccp.getSupervoxelToSegmentMap(mapping);
    supervoxel_to_segment.clear();
    for (const auto& item : mapping) {
        supervoxel_to_segment.push_back(item.first);
        supervoxel_to_segment.push_back(item.second);
    }
}

}  // namespace pclcompat
