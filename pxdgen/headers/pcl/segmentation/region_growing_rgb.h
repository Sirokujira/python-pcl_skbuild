// Mirror header: self-contained stand-in for
// <pcl/segmentation/region_growing_rgb.h>.
//
// RegionGrowing driven by colour instead of normals: neighbours join when
// their colour is within the point threshold, and regions whose mean
// colours differ by less than the region threshold are merged afterwards.
// Normals are only consulted when the normal test is switched on.
//
// Derives from RegionGrowing<PointT, NormalT> in PCL; the inherited
// methods used here are re-declared (flattened) as everywhere else, and
// the `NormalT = pcl::Normal` default has no Cython spelling, so the
// wrapper instantiates both arguments explicitly.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {

template <typename PointT, typename NormalT>
class RegionGrowingRGB {
public:
    RegionGrowingRGB();

    // Inherited entry points (PCLBase / RegionGrowing).
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

    float getPointColorThreshold() const;
    void setPointColorThreshold(float thresh);
    float getRegionColorThreshold() const;
    void setRegionColorThreshold(float thresh);
    float getDistanceThreshold() const;
    void setDistanceThreshold(float thresh);
    unsigned int getNumberOfRegionNeighbours() const;
    void setNumberOfRegionNeighbours(unsigned int nghbr_number);
    bool getNormalTestFlag() const;
    void setNormalTestFlag(bool value);
};

}  // namespace pcl
