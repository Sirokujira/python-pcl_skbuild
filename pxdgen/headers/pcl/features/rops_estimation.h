// Mirror header: self-contained stand-in for
// <pcl/features/rops_estimation.h>.
//
// Rotational Projection Statistics (Guo et al.): the local mesh around a
// point is rotated about its reference axes and projected; five
// statistics of each projection's distribution make the descriptor. Needs
// the triangles as well as the points.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class ROPSEstimation {
public:
    ROPSEstimation();

    // Inherited entry points (PCLBase / Feature).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setIndices(const std::shared_ptr<std::vector<int>>& indices);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setNumberOfPartitionBins(unsigned int number_of_bins);
    unsigned int getNumberOfPartitionBins() const;
    void setNumberOfRotations(unsigned int number_of_rotations);
    unsigned int getNumberOfRotations() const;
    void setSupportRadius(float support_radius);
    float getSupportRadius() const;
    void setTriangles(const std::vector<Vertices>& triangles);
};

}  // namespace pcl
