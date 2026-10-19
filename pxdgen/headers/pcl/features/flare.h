// Mirror header: self-contained stand-in for <pcl/features/flare.h>.
//
// FLARE local reference frames (Petrelli and Di Stefano): normal from a
// small radius, tangent axis from the point farthest from the tangent
// plane within a larger one. The signed-distance type parameter is left
// to its default (float).
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class FLARELocalReferenceFrameEstimation {
public:
    FLARELocalReferenceFrameEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setTangentRadius(float radius);
    float getTangentRadius() const;
    void setMarginThresh(float margin_thresh);
    float getMarginThresh() const;
    void setMinNeighboursForNormalAxis(int min_neighbors_for_normal_axis);
    int getMinNeighboursForNormalAxis() const;
    void setMinNeighboursForTangentAxis(int min_neighbors_for_tangent_axis);
    int getMinNeighboursForTangentAxis() const;
    const std::vector<float>& getSignedDistancesFromHighestPoints() const;
};

}  // namespace pcl
