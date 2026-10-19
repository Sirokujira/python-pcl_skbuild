// Mirror header: self-contained stand-in for <pcl/features/3dsc.h>.
//
// 3-D Shape Context (Frome et al.): a spherical histogram of the
// neighbourhood around each point's normal. The pxd is named
// shape_context_3d because a module name cannot start with a digit.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class ShapeContext3DEstimation {
public:
    explicit ShapeContext3DEstimation(bool random);

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setMinimalRadius(double radius);
    double getMinimalRadius();
    void setPointDensityRadius(double radius);
    double getPointDensityRadius();
};

}  // namespace pcl
