// Mirror header: self-contained stand-in for
// <pcl/segmentation/min_cut_segmentation.h>.
//
// Binary foreground/background segmentation by graph min-cut: mark some
// points as foreground, and it partitions the rest.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class MinCutSegmentation {
public:
    MinCutSegmentation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setForegroundPoints(
        const std::shared_ptr<PointCloud<PointT>>& foreground_points);
    void setSigma(double sigma);
    double getSigma() const;
    void setRadius(double radius);
    double getRadius() const;
    void setSourceWeight(double weight);
    double getSourceWeight() const;
    void setNumberOfNeighbours(unsigned int neighbour_number);
    unsigned int getNumberOfNeighbours() const;
    double getMaxFlow() const;

    void extract(std::vector<PointIndices>& clusters);
};

}  // namespace pcl
