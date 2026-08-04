// Mirror header: self-contained stand-in for
// <pcl/filters/random_sample.h>.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class RandomSample {
public:
    RandomSample();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setSample(unsigned int sample);
    unsigned int getSample();
    void setSeed(unsigned int seed);
    unsigned int getSeed();
};

}  // namespace pcl
