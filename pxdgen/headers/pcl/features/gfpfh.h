// Mirror header: self-contained stand-in for <pcl/features/gfpfh.h>.
//
// Global FPFH (Rusu et al.): one 16-bin descriptor for a whole cloud,
// from how the labelled surface classes sit next to each other in an
// octree. The labels come from a second cloud (FeatureFromLabels).
#pragma once

#include <cstdint>
#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointLT, typename PointOutT>
class GFPFHEstimation {
public:
    GFPFHEstimation();

    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputLabels(const std::shared_ptr<PointCloud<PointLT>>& labels);
    void compute(PointCloud<PointOutT>& output);

    void setOctreeLeafSize(double size);
    double getOctreeLeafSize();
    void setNumberOfClasses(std::uint32_t n);
    std::uint32_t getNumberOfClasses();
};

}  // namespace pcl
