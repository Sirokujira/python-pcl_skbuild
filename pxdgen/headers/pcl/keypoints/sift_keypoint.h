// Mirror header: self-contained stand-in for
// <pcl/keypoints/sift_keypoint.h>.
//
// SIFT over a 3-D scale space of the point "intensity" -- PCL provides
// the intensity for PointXYZRGB (luminance), PointXYZRGBA, PointXYZI and
// PointNormal (curvature); PointXYZ has none, so the wrapper takes a
// colour cloud. Output points carry the scale they were found at.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class SIFTKeypoint {
public:
    SIFTKeypoint();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setScales(float min_scale, int nr_octaves, int nr_scales_per_octave);
    void setMinimumContrast(float min_contrast);
};

}  // namespace pcl
