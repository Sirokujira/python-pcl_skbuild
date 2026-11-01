// Mirror header: self-contained stand-in for <pcl/features/normal_3d_omp.h>.
//
// NormalEstimation with OpenMP over the points. Same surface as
// features/normal_3d.h plus the thread count (0 = automatic).
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class NormalEstimationOMP {
public:
    NormalEstimationOMP();

    // Inherited entry points (PCLBase / Feature / NormalEstimation).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setKSearch(int k);
    int getKSearch() const;
    void setRadiusSearch(double radius);
    double getRadiusSearch() const;
    void compute(PointCloud<PointOutT>& output);
    void setViewPoint(float vpx, float vpy, float vpz);
    void useSensorOriginAsViewPoint();

    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
