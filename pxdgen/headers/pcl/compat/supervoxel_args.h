// Mirror header for src/pcl/compat/supervoxel_args.h: the map-and-multimap
// results of SupervoxelClustering, kept on the C++ side in a SupervoxelSet
// and read out as flat vectors, plus LCCP segmentation over them.
#pragma once

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/segmentation/supervoxel_clustering.h>

namespace pclcompat {

// Holds what extract() produced: the supervoxels and their adjacency.
template <typename PointT>
class SupervoxelSet {
public:
    SupervoxelSet();
};

template <typename PointT>
void supervoxelExtract(pcl::SupervoxelClustering<PointT>& clustering,
                       SupervoxelSet<PointT>& out);

// labels (k), centroids (3k: x y z), normals (4k: nx ny nz curvature),
// adjacency (2m: label pairs).
template <typename PointT>
void supervoxelSummary(const SupervoxelSet<PointT>& set,
                       std::vector<unsigned int>& labels,
                       std::vector<float>& centroids,
                       std::vector<float>& normals,
                       std::vector<unsigned int>& adjacency);

// The supervoxel label of every input point (0: none).
template <typename PointT>
void supervoxelPointLabels(const pcl::SupervoxelClustering<PointT>& clustering,
                           std::vector<unsigned int>& out);

// LCCP over the set: the segment label of every input point, and the
// supervoxel -> segment mapping as flat pairs.
template <typename PointT>
void lccpSegment(const SupervoxelSet<PointT>& set,
                 const pcl::SupervoxelClustering<PointT>& clustering,
                 float concavity_tolerance, bool smoothness_check,
                 float voxel_resolution, float seed_resolution,
                 float smoothness_threshold, bool sanity_check,
                 unsigned int k_factor, unsigned int min_segment_size,
                 std::vector<unsigned int>& point_labels,
                 std::vector<unsigned int>& supervoxel_to_segment);

}  // namespace pclcompat
