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

#include <pcl/octree/octree_pointcloud_density.h>
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

}  // namespace pclcompat
