// Mirror header: self-contained stand-in for
// <pcl/filters/covariance_sampling.h>.
//
// Covariance sampling (Gelfand et al.): picks the points that best
// constrain a point-to-plane ICP, so the sampled cloud keeps the
// geometry that makes registration well-conditioned.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointT, typename PointNT>
class CovarianceSampling {
public:
    CovarianceSampling();

    // Inherited entry points (PCLBase / Filter / FilterIndices).
    void setInputCloud(const std::shared_ptr<PointCloud<PointT>>& cloud);
    void filter(PointCloud<PointT>& output);
    void filter(std::vector<int>& indices);

    void setNumberOfSamples(unsigned int samples);
    unsigned int getNumberOfSamples() const;
    void setNormals(const std::shared_ptr<PointCloud<PointNT>>& normals);
    double computeConditionNumber();
};

}  // namespace pcl
