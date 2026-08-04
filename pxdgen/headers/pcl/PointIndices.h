// Mirror header: self-contained stand-in for <pcl/PointIndices.h>.
//
// PCL 1.11+ only: `indices` is pcl::Indices (std::vector<pcl::index_t>,
// index_t = int) and Ptr is std::shared_ptr. Both were boost-typed before
// 1.11 — see .claude/rules/pipeline.md for the version policy.
#pragma once

#include <memory>
#include <vector>

namespace pcl {

struct PointIndices {
    std::vector<int> indices;

    typedef std::shared_ptr<PointIndices> Ptr;
    typedef std::shared_ptr<const PointIndices> ConstPtr;
};

}  // namespace pcl
