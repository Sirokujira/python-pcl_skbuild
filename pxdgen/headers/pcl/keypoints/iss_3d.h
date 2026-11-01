// Mirror header: self-contained stand-in for <pcl/keypoints/iss_3d.h>.
//
// Intrinsic Shape Signatures keypoints: points whose neighbourhood
// scatter matrix has three well-separated eigenvalues, with non-maxima
// suppression. Normals are optional (needed only with a border radius).
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT, typename NormalT>
class ISSKeypoint3D {
public:
    ISSKeypoint3D();

    // Inherited entry points (PCLBase / Keypoint).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void compute(PointCloud<PointOutT>& output);

    void setSalientRadius(double salient_radius);
    void setNonMaxRadius(double non_max_radius);
    void setNormalRadius(double normal_radius);
    void setBorderRadius(double border_radius);
    void setThreshold21(double gamma_21);
    void setThreshold32(double gamma_32);
    void setMinNeighbors(int min_neighbors);
    void setNormals(const std::shared_ptr<PointCloud<NormalT>>& normals);
    void setAngleThreshold(float angle);
    void setNumberOfThreads(unsigned int nr_threads);
};

}  // namespace pcl
