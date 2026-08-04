// Mirror header: self-contained stand-in for <pcl/filters/passthrough.h>.
// See filters/voxel_grid.h for why inherited methods are flattened.
#pragma once

#include <memory>
#include <string>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class PassThrough {
public:
    PassThrough();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);
    bool getNegative() const;
    void setKeepOrganized(bool keep_organized);
    bool getKeepOrganized() const;

    void setFilterFieldName(const std::string& field_name);
    std::string getFilterFieldName() const;
    void setFilterLimits(const float& limit_min, const float& limit_max);
};

}  // namespace pcl
