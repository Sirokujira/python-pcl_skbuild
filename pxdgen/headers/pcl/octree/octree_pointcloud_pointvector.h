// Mirror header: self-contained stand-in for
// <pcl/octree/octree_pointcloud_pointvector.h>.
//
// An octree whose leaves keep the indices of every point in them. Only the point type is declared; C++ fills the
// container defaults.
#pragma once

#include <cstddef>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {
namespace octree {

template <typename PointT>
class OctreePointCloudPointVector {
public:
    explicit OctreePointCloudPointVector(double resolution);

    // Inherited entry points (OctreePointCloud / OctreeBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void addPointsFromInputCloud();
    void defineBoundingBox(double min_x, double min_y, double min_z,
                           double max_x, double max_y, double max_z);
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
