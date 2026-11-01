// Mirror header: self-contained stand-in for <pcl/segmentation/approximate_progressive_morphological_filter.h>.
//
// ProgressiveMorphologicalFilter on a height grid instead of per point:
// the same settings, much faster, slightly coarser.
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT>
class ApproximateProgressiveMorphologicalFilter {
public:
    ApproximateProgressiveMorphologicalFilter();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);

    void setMaxWindowSize(int max_window_size);
    int getMaxWindowSize() const;
    void setSlope(float slope);
    float getSlope() const;
    void setMaxDistance(float max_distance);
    float getMaxDistance() const;
    void setInitialDistance(float initial_distance);
    float getInitialDistance() const;
    void setCellSize(float cell_size);
    float getCellSize() const;
    void setBase(float base);
    float getBase() const;
    void setExponential(bool exponential);
    bool getExponential() const;
    void setNumberOfThreads(unsigned int nr_threads);
    void extract(std::vector<int>& ground);
};

}  // namespace pcl
