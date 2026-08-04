// Mirror header: self-contained stand-in for
// <pcl/octree/octree_pointcloud_changedetector.h>.
//
// Spatial change detection between two clouds: fill the octree from the
// first cloud, switchBuffers(), fill it from the second, then ask which
// voxels are new. See filters/voxel_grid.h for the flattening rationale.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {
namespace octree {

template <typename PointT>
class OctreePointCloudChangeDetector {
public:
    OctreePointCloudChangeDetector(double resolution);

    // Inherited entry points (OctreePointCloud / Octree2BufBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void addPointsFromInputCloud();
    void deleteTree();
    void switchBuffers();
    double getResolution() const;
    unsigned int getTreeDepth() const;

    int getPointIndicesFromNewVoxels(std::vector<int>& indices,
                                     int min_points_per_leaf = 0);
};

}  // namespace octree
}  // namespace pcl
