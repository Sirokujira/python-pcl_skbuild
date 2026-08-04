// Mirror header: self-contained stand-in for <pcl/filters/crop_box.h>.
//
// setMin/setMax take an Eigen::Vector4f and setTranslation/setRotation an
// Eigen::Vector3f, none of which a mirror header may name — they are
// reached through pcl/compat/eigen_args.h, which takes plain floats.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class CropBox {
public:
    CropBox();
    CropBox(bool extract_removed_indices);

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);
    bool getNegative() const;
    void setKeepOrganized(bool keep_organized);
};

}  // namespace pcl
