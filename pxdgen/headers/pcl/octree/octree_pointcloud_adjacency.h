// Mirror header: self-contained stand-in for
// <pcl/octree/octree_pointcloud_adjacency.h>.
//
// An octree whose leaves know their 26-neighbourhood, the structure
// SupervoxelClustering builds on. The adjacency comes out through
// pclcompat::adjacencyGraph, occlusion through pclcompat::adjacencyOccluded.
// No defineBoundingBox: addPointsFromInputCloud always resets the box to
// the cloud's extent. Only the point type is declared; C++ fills the
// container defaults.
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {
namespace octree {

template <typename PointT>
class OctreePointCloudAdjacency {
public:
    explicit OctreePointCloudAdjacency(double resolution);

    // Inherited entry points (OctreePointCloud / OctreeBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void addPointsFromInputCloud();
    void deleteTree();
    double getResolution() const;
    unsigned int getTreeDepth() const;
    std::size_t getLeafCount() const;
    bool isVoxelOccupiedAtPoint(double point_x_arg, double point_y_arg,
                                double point_z_arg) const;
    void deleteVoxelAtPoint(const PointT& point_arg);
};

}  // namespace octree
}  // namespace pcl
