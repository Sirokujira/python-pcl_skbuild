// Octree calls Cython cannot state directly.
//
// getIntersectedVoxelCenters / getIntersectedVoxelIndices take the ray
// as two Eigen::Vector3f by value, and getVoxelCentroids /
// getOccupiedVoxelCenters fill an `AlignedPointTVector` -- a std::vector
// with Eigen's aligned allocator, which is a different C++ type from the
// std::vector Cython can name. Each function here takes plain floats
// and fills a plain vector. Templates on the octree type: the mirror
// declares one overload per class the wrappers use, and C++ deduces.
//
// VoxelCentroidOctree exists because PCL 1.14's header-only
// OctreePointCloudVoxelCentroid::getVoxelCentroidAtPoint does not
// compile: it calls genOctreeKeyforPoint, a member of a dependent base,
// without `this->`, which two-phase lookup rejects ("not declared in
// this scope ... use this->genOctreeKeyforPoint instead"). PCL never
// instantiates the class itself ("Don't precompile this octree type"),
// so nothing upstream notices. The subclass does the same lookup with
// the qualification, and nothing here ever calls the broken member.
#pragma once

#include <vector>

#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class VoxelCentroidOctree
    : public pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZ> {
public:
    explicit VoxelCentroidOctree(double resolution)
        : pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZ>(resolution) {}

    bool centroidAtPoint(const pcl::PointXYZ& point,
                         pcl::PointXYZ& centroid) const {
        pcl::octree::OctreeKey key;
        this->genOctreeKeyforPoint(point, key);
        const LeafContainer* leaf = this->findLeaf(key);
        if (leaf != nullptr) {
            leaf->getCentroid(centroid);
        }
        return leaf != nullptr;
    }
};

// PCL's ray casting applies `max_voxel_count` per subtree: every
// recursion level compares its OWN count with the global maximum, so the
// result can hold more voxels than asked for (measured: 3 for a maximum
// of 2). The list is in ray order, so the first `max_count` entries are
// exactly the voxels the limit means; the centres are truncated to them,
// and the limited index query is answered from those voxels.
template <typename Octree>
inline void intersectedVoxelCenters(const Octree& octree,
                                    const float* origin3,
                                    const float* direction3,
                                    unsigned int max_count,
                                    std::vector<pcl::PointXYZ>& out) {
    typename Octree::AlignedPointTVector centers;
    octree.getIntersectedVoxelCenters(
        Eigen::Vector3f(origin3[0], origin3[1], origin3[2]),
        Eigen::Vector3f(direction3[0], direction3[1], direction3[2]),
        centers, max_count);
    if (max_count > 0 && centers.size() > max_count) {
        centers.resize(max_count);
    }
    out.assign(centers.begin(), centers.end());
}

template <typename Octree>
inline void intersectedVoxelIndices(Octree& octree, const float* origin3,
                                    const float* direction3,
                                    unsigned int max_count,
                                    std::vector<int>& out) {
    out.clear();
    if (max_count == 0) {
        pcl::Indices indices;
        octree.getIntersectedVoxelIndices(
            Eigen::Vector3f(origin3[0], origin3[1], origin3[2]),
            Eigen::Vector3f(direction3[0], direction3[1], direction3[2]),
            indices, 0);
        out.assign(indices.begin(), indices.end());
        return;
    }
    std::vector<pcl::PointXYZ> centers;
    intersectedVoxelCenters(octree, origin3, direction3, max_count, centers);
    for (const auto& center : centers) {
        pcl::Indices in_voxel;
        octree.voxelSearch(center, in_voxel);
        out.insert(out.end(), in_voxel.begin(), in_voxel.end());
    }
}

template <typename Octree>
inline void voxelCentroids(const Octree& octree,
                           std::vector<pcl::PointXYZ>& out) {
    typename Octree::AlignedPointTVector centroids;
    octree.getVoxelCentroids(centroids);
    out.assign(centroids.begin(), centroids.end());
}

template <typename Octree>
inline void occupiedVoxelCentersOf(const Octree& octree,
                                   std::vector<pcl::PointXYZ>& out) {
    typename Octree::AlignedPointTVector centers;
    octree.getOccupiedVoxelCenters(centers);
    out.assign(centers.begin(), centers.end());
}

}  // namespace pclcompat
