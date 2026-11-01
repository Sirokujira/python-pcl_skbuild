// Mirror header: self-contained stand-in for
// <pcl/segmentation/region_growing.h>.
//
// Grows smooth regions outwards from low-curvature seeds: a neighbour
// joins when its normal deviates by less than the smoothness threshold,
// and growth stops at points whose curvature exceeds the curvature
// threshold. Needs normals (setInputNormals).
//
// setSearchMethod takes a search::Search<PointT>::Ptr, which would drag
// PCL's whole search hierarchy in; the class builds a default KdTree when
// none is set, so it is left out (same choice as features/normal_3d.h).
// getColoredCloud returns PointCloud<PointXYZRGB>::Ptr, hence the
// point_types include. NormalPtr is the NON-const Ptr in PCL itself.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {

template <typename PointT, typename NormalT>
class RegionGrowing {
public:
    RegionGrowing();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    int getMinClusterSize();
    void setMinClusterSize(int min_cluster_size);
    int getMaxClusterSize();
    void setMaxClusterSize(int max_cluster_size);
    bool getSmoothModeFlag() const;
    void setSmoothModeFlag(bool value);
    bool getCurvatureTestFlag() const;
    void setCurvatureTestFlag(bool value);
    bool getResidualTestFlag() const;
    void setResidualTestFlag(bool value);
    float getSmoothnessThreshold() const;
    void setSmoothnessThreshold(float theta);
    float getResidualThreshold() const;
    void setResidualThreshold(float residual);
    float getCurvatureThreshold() const;
    void setCurvatureThreshold(float curvature);
    unsigned int getNumberOfNeighbours() const;
    void setNumberOfNeighbours(unsigned int neighbour_number);
    std::shared_ptr<PointCloud<NormalT>> getInputNormals() const;
    void setInputNormals(const std::shared_ptr<PointCloud<NormalT>>& norm);

    void extract(std::vector<PointIndices>& clusters);
    void getSegmentFromPoint(int index, PointIndices& cluster);
    std::shared_ptr<PointCloud<PointXYZRGB>> getColoredCloud();
};

}  // namespace pcl
