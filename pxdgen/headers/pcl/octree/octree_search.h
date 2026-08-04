// Mirror header: self-contained stand-in for
// <pcl/octree/octree_search.h>.
//
// getOccupiedVoxelCenters is NOT declared here: it takes an
// `AlignedPointTVector`, i.e. std::vector<PointT,
// Eigen::aligned_allocator<PointT>>, which is a different C++ type from
// the std::vector Cython can name -- passing one for the other would not
// compile. It is reached through pcl/compat/octree_voxels.h instead.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {
namespace octree {

template <typename PointT>
class OctreePointCloudSearch {
public:
    OctreePointCloudSearch(double resolution);

    // Inherited entry points (OctreePointCloud / OctreeBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void addPointsFromInputCloud();
    void defineBoundingBox();
    void defineBoundingBox(double min_x, double min_y, double min_z,
                           double max_x, double max_y, double max_z);
    void deleteTree();
    double getResolution() const;
    unsigned int getTreeDepth() const;

    bool voxelSearch(const PointT& point, std::vector<int>& point_idx_data);

    int nearestKSearch(const PointT& p_q, int k,
                       std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances);

    int radiusSearch(const PointT& p_q, double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn = 0);

    void approxNearestSearch(const PointT& p_q, int& result_index,
                             float& sqr_distance);
};

}  // namespace octree
}  // namespace pcl
