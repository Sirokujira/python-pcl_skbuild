// pcl::outofcore::OutofcoreOctreeBase (the disk-backed octree for clouds
// larger than memory) behind one Cython-expressible class. Built only
// where PCL has outofcore, which PCL builds only with its VTK-based
// visualization module; its templates include VTK headers.
//
// Two faults absorbed, both measured in plain C++ against 1.14:
//
// - A freshly built tree answers every query with nothing: points sit in
//   each node's write buffer, and OutofcoreOctreeDiskContainer::readRange
//   reads only the node's file on disk (ignoring its start / count too).
//   20,000 points added, a box holding 5,030 of them returned 0; after
//   the tree was closed and reopened, 5,030.
// - The way to flush, OutofcoreOctreeBase::flushToDisk(), calls
//   root_node_->flushToDisk(), which the node class does not have; it
//   has never compiled (PCL never instantiates it). The destructor
//   flushes correctly, so OutofcoreTree closes and reopens the tree
//   after every change.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pcl/outofcore/outofcore.h>
#include <pcl/outofcore/outofcore_impl.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class OutofcoreTree {
    using Tree = pcl::outofcore::OutofcoreOctreeBase<>;

public:
    /// Create a tree of `depth` levels over the box, rooted at `path`
    /// (a ".oct_idx" file in a directory of its own).
    void create(const std::string& path, const double* min3, const double* max3,
                unsigned int depth) {
        path_ = path;
        tree_ = std::make_unique<Tree>(depth, Eigen::Vector3d(min3[0], min3[1], min3[2]),
                                       Eigen::Vector3d(max3[0], max3[1], max3[2]),
                                       boost::filesystem::path(path), "ECEF");
        reopen();
    }

    void open(const std::string& path) {
        path_ = path;
        tree_ = std::make_unique<Tree>(boost::filesystem::path(path), false);
    }

    /// Points inside the tree's box are stored; returns how many.
    std::uint64_t add(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud) {
        const std::uint64_t added = tree_->addPointCloud(cloud);
        reopen();
        return added;
    }

    /// Subsample each level from the one below (PCL's level of detail).
    void buildLOD() {
        tree_->buildLOD();
        reopen();
    }

    void query(const double* min3, const double* max3, unsigned int depth,
               pcl::PointCloud<pcl::PointXYZ>& out) const {
        Tree::AlignedPointTVector points;
        tree_->queryBBIncludes(Eigen::Vector3d(min3[0], min3[1], min3[2]),
                               Eigen::Vector3d(max3[0], max3[1], max3[2]), depth, points);
        out.clear();
        for (const auto& p : points) {
            out.push_back(p);
        }
    }

    void voxelCentres(unsigned int depth, std::vector<double>& out) const {
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> centres;
        tree_->getOccupiedVoxelCenters(centres, depth);
        out.clear();
        for (const auto& c : centres) {
            out.insert(out.end(), {c[0], c[1], c[2]});
        }
    }

    unsigned int depth() const { return static_cast<unsigned int>(tree_->getDepth()); }
    double voxelSideLength(unsigned int depth) const {
        return tree_->getVoxelSideLength(depth);
    }
    std::uint64_t pointsAtDepth(unsigned int depth) const {
        return tree_->getNumPointsAtDepth(depth);
    }
    void boundingBox(double* min3, double* max3) const {
        Eigen::Vector3d lo, hi;
        tree_->getBoundingBox(lo, hi);
        for (int i = 0; i < 3; ++i) {
            min3[i] = lo[i];
            max3[i] = hi[i];
        }
    }

private:
    void reopen() {
        tree_.reset();  // the destructor flushes every node and the index
        tree_ = std::make_unique<Tree>(boost::filesystem::path(path_), false);
    }

    std::string path_;
    std::unique_ptr<Tree> tree_;
};

}  // namespace pclcompat
