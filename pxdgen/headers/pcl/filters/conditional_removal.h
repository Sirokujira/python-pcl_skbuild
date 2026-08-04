// Mirror header: self-contained stand-in for
// <pcl/filters/conditional_removal.h>.
//
// Only the two types the wrapper names appear here. The condition tree
// they hold is built entirely through pcl/compat/condition_args.h: its
// nodes are shared_ptrs to abstract bases, and Cython cannot perform the
// derived-to-base shared_ptr conversions that assembling one requires.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ConditionAnd {
public:
    ConditionAnd();
};

template <typename PointT>
class ConditionalRemoval {
public:
    ConditionalRemoval();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setKeepOrganized(bool keep_organized);
    bool getKeepOrganized() const;
    void setUserFilterValue(float value);
};

}  // namespace pcl
