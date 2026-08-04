// Mirror header: self-contained stand-in for <pcl/surface/mls.h>.
//
// Moving Least Squares: smooths a cloud by fitting a polynomial surface
// to each neighbourhood, and optionally estimates normals while doing it
// (which is why the output point type differs from the input one).
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT, typename PointOutT>
class MovingLeastSquares {
public:
    MovingLeastSquares();

    // Inherited entry point (PCLBase).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);

    void setSearchRadius(double radius);
    double getSearchRadius() const;
    void setPolynomialOrder(int order);
    int getPolynomialOrder() const;
    void setComputeNormals(bool compute_normals);
    void setSqrGaussParam(double sqr_gauss_param);
    double getSqrGaussParam() const;
    void setNumberOfThreads(unsigned int threads);

    void process(PointCloud<PointOutT>& output);
};

}  // namespace pcl
