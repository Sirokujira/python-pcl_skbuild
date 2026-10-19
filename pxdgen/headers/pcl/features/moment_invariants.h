// Mirror header: self-contained stand-in for <pcl/features/moment_invariants.h>.
//
// The three moment invariants of each point's neighbourhood. Needs a
// neighbourhood size.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class MomentInvariantsEstimation {
public:
    MomentInvariantsEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setKSearch(int k);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);
};

}  // namespace pcl
