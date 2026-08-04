// Mirror header: self-contained stand-in for
// <pcl/filters/extract_indices.h>.
//
// The other half of segmentation: SACSegmentation and
// EuclideanClusterExtraction hand back indices, and this is what turns
// them into a cloud.
//
// See filters/voxel_grid.h for why the inherited methods are flattened
// and why const is dropped inside the shared_ptr.
#pragma once

#include <memory>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ExtractIndices {
public:
    ExtractIndices();
    ExtractIndices(bool extract_removed_indices);

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setIndices(const std::shared_ptr<PointIndices>& indices);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);
    bool getNegative() const;
    void setKeepOrganized(bool keep_organized);
    void setUserFilterValue(float value);
};

}  // namespace pcl
