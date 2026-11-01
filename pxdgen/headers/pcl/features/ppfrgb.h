// Mirror header: self-contained stand-in for <pcl/features/ppfrgb.h>.
//
// PPF plus three colour ratios. PCL precompiles only the Region variant,
// so the pxd is extern-from pcl/compat/ppfrgb.h, which includes the impl.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class PPFRGBEstimation {
public:
    PPFRGBEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
