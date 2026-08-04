// Mirror header: self-contained stand-in for <pcl/filters/voxel_grid.h>.
//
// FLATTENED HIERARCHY -- read before adding a class here.
// The real VoxelGrid derives from Filter<PointT> -> PCLBase<PointT>, and
// setInputCloud/filter come from those bases. A pxd only has to declare
// that a NAME exists on a type; C++ resolves it through the real base
// classes at compile time. So each concrete filter re-declares the
// inherited entry points it needs instead of mirroring the whole
// PCLBase/Filter/FilterIndices chain. That keeps the mirror headers small
// and avoids modelling PCL's template inheritance in Cython.
//
// setInputCloud takes shared_ptr<PointCloud<PointT>> (non-const element)
// while the real signature takes shared_ptr<const PointCloud<PointT>>:
// the implicit shared_ptr<T> -> shared_ptr<const T> conversion makes the
// call compile, and it spares the pxd a `const` inside template
// arguments, which Cython handles poorly.
#pragma once

#include <memory>
#include <string>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class VoxelGrid {
public:
    VoxelGrid();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setLeafSize(float lx, float ly, float lz);
    void setDownsampleAllData(bool downsample);
    bool getDownsampleAllData() const;
    void setMinimumPointsNumberPerVoxel(unsigned int min_points_per_voxel);
    unsigned int getMinimumPointsNumberPerVoxel() const;
    void setSaveLeafLayout(bool save_leaf_layout);
    bool getSaveLeafLayout() const;
    void setFilterFieldName(const std::string& field_name);
    std::string getFilterFieldName() const;
    void setFilterLimits(const double& limit_min, const double& limit_max);
    void setFilterLimitsNegative(const bool limit_negative);
    bool getFilterLimitsNegative() const;
};

}  // namespace pcl
