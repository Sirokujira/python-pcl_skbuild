// Mirror header: self-contained stand-in for <pcl/features/esf.h>.
//
// Ensemble of Shape Functions: one 640-bin descriptor for the whole
// cloud, built from distances, angles and areas of randomly sampled
// point pairs and triples. Needs neither normals nor a search size (the
// constructor sets k = 5 for the base class's benefit).
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class ESFEstimation {
public:
    ESFEstimation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);

    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
