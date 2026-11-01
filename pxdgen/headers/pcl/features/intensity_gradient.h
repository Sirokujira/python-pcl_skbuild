// Mirror header: self-contained stand-in for
// <pcl/features/intensity_gradient.h>.
//
// The 3-D gradient of the intensity field around each point, projected
// onto its tangent plane. The intensity-selector parameter is left to its
// default (the point's `intensity` field).
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class IntensityGradientEstimation {
public:
    IntensityGradientEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
