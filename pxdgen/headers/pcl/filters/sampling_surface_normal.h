// Mirror header: self-contained stand-in for <pcl/filters/sampling_surface_normal.h>.
//
// Splits the cloud into cells of `sample` points, keeps a `ratio` of each
// and writes the cell's plane normal into the kept points -- so PointT
// must carry normal fields (PointNormal here).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class SamplingSurfaceNormal {
public:
    SamplingSurfaceNormal();

    // Inherited entry points (PCLBase / Filter).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);

    void setSample(unsigned int sample);
    unsigned int getSample() const;
    void setSeed(unsigned int seed);
    unsigned int getSeed() const;
    void setRatio(float ratio);
    float getRatio() const;
};

}  // namespace pcl
