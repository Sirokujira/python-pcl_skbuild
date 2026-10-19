// Mirror header: self-contained stand-in for
// <pcl/filters/farthest_point_sampling.h>.
//
// Farthest point sampling: start from a random point, then repeatedly
// add the point farthest from everything picked so far -- an even
// spatial cover of the cloud in `sample` points.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class FarthestPointSampling {
public:
    FarthestPointSampling();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void filter(std::vector<int>& indices);

    void setSample(std::size_t sample_size);
    int getSample() const;
    void setSeed(unsigned int seed);
    unsigned int getSeed() const;
};

}  // namespace pcl
