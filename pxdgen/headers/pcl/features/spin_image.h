// Mirror header: self-contained stand-in for <pcl/features/spin_image.h>.
//
// Spin images: a 2-D histogram of each point's neighbourhood in the
// (radial, elevation) plane around its normal. The output type is
// Histogram<153> (image width 8), reached through compat/feature_types.h.
// Needs normals and a neighbourhood size.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class SpinImageEstimation {
public:
    SpinImageEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setImageWidth(unsigned int bin_count);
    void setSupportAngle(double support_angle_cos);
    void setMinPointCountInNeighbourhood(unsigned int min_pts_neighb);
    void setAngularDomain(bool is_angular);
    void setRadialStructure(bool is_radial);
};

}  // namespace pcl
