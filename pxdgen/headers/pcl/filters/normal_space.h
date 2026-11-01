// Mirror header: self-contained stand-in for <pcl/filters/normal_space.h>.
//
// Subsamples so that the normals of the kept points cover normal space
// evenly (bins over the unit sphere), which keeps the flat parts from
// drowning the edges. Needs normals.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename NormalT>
class NormalSpaceSampling {
public:
    NormalSpaceSampling();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void setNegative(bool negative);

    void setSample(unsigned int sample);
    unsigned int getSample() const;
    void setSeed(unsigned int seed);
    unsigned int getSeed() const;
    void setBins(unsigned int binsx, unsigned int binsy, unsigned int binsz);
    void setNormals(const std::shared_ptr<PointCloud<NormalT>>& normals);
};

}  // namespace pcl
