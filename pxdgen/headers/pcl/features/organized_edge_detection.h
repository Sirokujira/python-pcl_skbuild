// Mirror header: self-contained stand-in for
// <pcl/features/organized_edge_detection.h>.
//
// Edges in an organized cloud: depth discontinuities (occluding and
// occluded sides, NaN boundaries), high curvature from the normals, and
// Canny edges in the colour image. compute() returns a per-pixel label
// cloud plus one index list per edge type. The EdgeType enum constants
// come through pcl/compat/feature_types.h.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointLT>
class OrganizedEdgeBase {
public:
    OrganizedEdgeBase();

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void compute(PointCloud<PointLT>& labels,
                 std::vector<PointIndices>& label_indices) const;

    void setDepthDisconThreshold(const float th);
    float getDepthDisconThreshold() const;
    void setMaxSearchNeighbors(const int max_dist);
    int getMaxSearchNeighbors() const;
    void setEdgeType(int edge_types);
    int getEdgeType() const;
};

template <typename PointT, typename PointNT, typename PointLT>
class OrganizedEdgeFromNormals {
public:
    OrganizedEdgeFromNormals();

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointLT>& labels,
                 std::vector<PointIndices>& label_indices) const;

    void setDepthDisconThreshold(const float th);
    void setMaxSearchNeighbors(const int max_dist);
    void setEdgeType(int edge_types);
    int getEdgeType() const;
    void setHCCannyLowThreshold(const float th);
    void setHCCannyHighThreshold(const float th);
};

template <typename PointT, typename PointLT>
class OrganizedEdgeFromRGB {
public:
    OrganizedEdgeFromRGB();

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void compute(PointCloud<PointLT>& labels,
                 std::vector<PointIndices>& label_indices) const;

    void setDepthDisconThreshold(const float th);
    void setMaxSearchNeighbors(const int max_dist);
    void setEdgeType(int edge_types);
    int getEdgeType() const;
    void setRGBCannyLowThreshold(const float th);
    void setRGBCannyHighThreshold(const float th);
};

template <typename PointT, typename PointNT, typename PointLT>
class OrganizedEdgeFromRGBNormals {
public:
    OrganizedEdgeFromRGBNormals();

    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void compute(PointCloud<PointLT>& labels,
                 std::vector<PointIndices>& label_indices) const;

    void setDepthDisconThreshold(const float th);
    void setMaxSearchNeighbors(const int max_dist);
    void setEdgeType(int edge_types);
    int getEdgeType() const;
    void setHCCannyLowThreshold(const float th);
    void setHCCannyHighThreshold(const float th);
    void setRGBCannyLowThreshold(const float th);
    void setRGBCannyHighThreshold(const float th);
};

}  // namespace pcl
