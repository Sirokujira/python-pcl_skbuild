// Mirror header: self-contained stand-in for <pcl/features/usc.h>.
//
// Unique Shape Context (Tombari et al.): 3DSC computed in a repeatable
// local reference frame, so no azimuth ambiguity and no normals needed.
// The reference-frame type parameter is left to its default.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class UniqueShapeContext {
public:
    UniqueShapeContext();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setMinimalRadius(double radius);
    double getMinimalRadius();
    void setPointDensityRadius(double radius);
    double getPointDensityRadius();
    void setLocalRadius(double radius);
    double getLocalRadius();
};

}  // namespace pcl
