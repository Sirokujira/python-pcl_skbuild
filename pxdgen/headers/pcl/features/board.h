// Mirror header: self-contained stand-in for <pcl/features/board.h>.
//
// BOARD local reference frames (Petrelli and Di Stefano): the normal plus
// a tangent axis found from the neighbourhood's farthest-from-plane
// point, robust at the edge of a partial view (the hole detection).
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointNT, typename PointOutT>
class BOARDLocalReferenceFrameEstimation {
public:
    BOARDLocalReferenceFrameEstimation();

    // Inherited entry points (PCLBase / Feature / FeatureFromNormals).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void setInputNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    void setRadiusSearch(double radius);
    void compute(PointCloud<PointOutT>& output);

    void setTangentRadius(float radius);
    float getTangentRadius() const;
    void setFindHoles(bool find_holes);
    bool getFindHoles() const;
    void setMarginThresh(float margin_thresh);
    float getMarginThresh() const;
    void setCheckMarginArraySize(int size);
    int getCheckMarginArraySize() const;
    void setHoleSizeProbThresh(float prob_thresh);
    float getHoleSizeProbThresh() const;
    void setSteepThresh(float steep_thresh);
    float getSteepThresh() const;
};

}  // namespace pcl
