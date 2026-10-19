// Mirror header: self-contained stand-in for
// <pcl/segmentation/supervoxel_clustering.h>.
//
// Supervoxels: compact, roughly equal clusters that respect colour,
// position and normal continuity. extract() hands back a std::map of
// Supervoxel objects and the adjacency a std::multimap, neither of which
// a pxd can hold, so those live in compat/supervoxel_args.h; this
// declares the settings. PCL compiles the class for PointXYZ,
// PointXYZRGB and PointXYZRGBA only.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {

template <typename PointT>
class SupervoxelClustering {
public:
    SupervoxelClustering(float voxel_resolution, float seed_resolution);

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setNormalCloud(const std::shared_ptr<PointCloud<Normal>>& normal_cloud);
    void setVoxelResolution(float resolution);
    float getVoxelResolution() const;
    void setSeedResolution(float seed_resolution);
    float getSeedResolution() const;
    void setColorImportance(float val);
    void setSpatialImportance(float val);
    void setNormalImportance(float val);
    void setUseSingleCameraTransform(bool val);
    int getMaxLabel() const;
};

}  // namespace pcl
