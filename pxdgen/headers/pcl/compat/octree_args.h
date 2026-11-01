// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/octree_args.h.
//
// Ray casting through an octree takes two Eigen::Vector3f by value, and
// every "list the voxels" call fills a vector with Eigen's aligned
// allocator, a type Cython cannot name. Origins and directions cross as
// float[3]; results as plain vectors.
//
// VoxelCentroidOctree stands in for pcl::octree::OctreePointCloudVoxelCentroid
// <PointXYZ>, whose getVoxelCentroidAtPoint does not compile (see the
// real header); the inherited entry points are flattened in.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <pcl/octree/octree_pointcloud_adjacency.h>
#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/octree_pointcloud_singlepoint.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class VoxelCentroidOctree {
public:
    explicit VoxelCentroidOctree(double resolution);

    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);
    void addPointsFromInputCloud();
    void defineBoundingBox(double min_x, double min_y, double min_z,
                           double max_x, double max_y, double max_z);
    void deleteTree();
    double getResolution() const;
    unsigned int getTreeDepth() const;
    std::size_t getLeafCount() const;
    bool isVoxelOccupiedAtPoint(double point_x_arg, double point_y_arg,
                                double point_z_arg) const;

    bool centroidAtPoint(const pcl::PointXYZ& point,
                         pcl::PointXYZ& centroid) const;
};

// Voxels a ray from `origin` along `direction` passes through, nearest
// first; max_count 0 means all of them.
void intersectedVoxelCenters(
    const pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>& octree,
    const float* origin3, const float* direction3, unsigned int max_count,
    std::vector<pcl::PointXYZ>& out);
void intersectedVoxelIndices(
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>& octree,
    const float* origin3, const float* direction3, unsigned int max_count,
    std::vector<int>& out);

void voxelCentroids(const VoxelCentroidOctree& octree,
                    std::vector<pcl::PointXYZ>& out);

void occupiedVoxelCentersOf(const VoxelCentroidOctree& octree,
                            std::vector<pcl::PointXYZ>& out);
void occupiedVoxelCentersOf(
    const pcl::octree::OctreePointCloudDensity<pcl::PointXYZ>& octree,
    std::vector<pcl::PointXYZ>& out);

// Leaf centres (3 floats each), and the point indices of every leaf:
// counts[i] for leaf i, in order from flat. One overload per octree.
void leafCenters(pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ>& octree,
                 std::vector<float>& centers);
void leafIndices(
    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ>& octree,
    std::vector<int>& flat, std::vector<int>& counts,
    std::vector<float>& centers);
void leafIndices(
    pcl::octree::OctreePointCloudSinglePoint<pcl::PointXYZ>& octree,
    std::vector<int>& flat, std::vector<int>& counts,
    std::vector<float>& centers);

// Leaf centres and point counts, and each neighbouring pair once.
void adjacencyGraph(
    pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZ>& octree,
    std::vector<float>& centers, std::vector<int>& counts,
    std::vector<int>& edges);

// testForOcclusion that stops at the octree's bounding box (PCL's own
// wraps its keys outside it and reports false occlusions).
bool adjacencyOccluded(
    const pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZ>& octree,
    const pcl::PointXYZ& point, const pcl::PointXYZ& camera);

}  // namespace pclcompat
