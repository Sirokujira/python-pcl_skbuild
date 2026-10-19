// Mirror header: self-contained stand-in for <pcl/segmentation/seeded_hue_segmentation.h>.
//
// Grows regions of similar hue outwards from seed points on a colour
// cloud. Not a template in PCL: it works on PointXYZRGB only.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl {

class SeededHueSegmentation {
public:
    SeededHueSegmentation();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointXYZRGB>>& cloud);

    void setClusterTolerance(double tolerance);
    double getClusterTolerance() const;
    void setDeltaHue(float delta_hue);
    float getDeltaHue() const;
    void segment(PointIndices& indices_in, PointIndices& indices_out);
};

}  // namespace pcl
