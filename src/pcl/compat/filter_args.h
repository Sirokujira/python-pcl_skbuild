// Filters whose PCL API Cython cannot state directly; see the mirror
// header for each function's contract.
//
// - NormalRefinement needs every point's neighbour indices and squared
//   distances up front; they are searched here with a kd-tree.
// - VoxelGridLabel works on PointXYZRGBL only; the cloud is converted.
// - VoxelGridCovariance's results are Eigen vectors and matrices inside
//   a std::map of leaves.
// - VoxelGridOcclusionEstimation reads the sensor origin from the cloud
//   and answers in Eigen::Vector3i grid coordinates.
// - The clippers take Eigen planes and boxes, Convolution3D a kernel by
//   value, filters::Convolution an Eigen::ArrayXf, and the morphological
//   operator an enum.
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// convolution_3d.h names pcl::search::KdTree without including it, so
// the search header has to come first.
#include <pcl/search/kdtree.h>

#include <pcl/filters/box_clipper3D.h>
#include <pcl/filters/convolution.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/normal_refinement.h>
#include <pcl/filters/plane_clipper3D.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/filters/voxel_grid_label.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#if PCL_VERSION_COMPARE(>=, 1, 13, 0)
#include <pcl/filters/farthest_point_sampling.h>
#endif

namespace pclcompat {

// pcl::FarthestPointSampling arrived in PCL 1.13. This class has the
// same calls on every supported version; before 1.13 it keeps the
// settings and filter() throws, so the Python class exists everywhere
// and raises a RuntimeError naming the version where PCL lacks it.
class FarthestPointSampler {
public:
#if PCL_VERSION_COMPARE(>=, 1, 13, 0)
    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud) {
        fps_.setInputCloud(cloud);
    }
    void setSample(std::size_t sample) { fps_.setSample(sample); }
    std::size_t getSample() const { return fps_.getSample(); }
    void setSeed(unsigned int seed) { fps_.setSeed(seed); }
    unsigned int getSeed() const { return fps_.getSeed(); }
    void filter(pcl::PointCloud<pcl::PointXYZ>& out) { fps_.filter(out); }
    void filter(std::vector<int>& indices) {
        pcl::Indices picked;
        fps_.filter(picked);
        indices.assign(picked.begin(), picked.end());
    }

private:
    pcl::FarthestPointSampling<pcl::PointXYZ> fps_;
#else
    void setInputCloud(
        const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>&) {}
    void setSample(std::size_t sample) { sample_ = sample; }
    std::size_t getSample() const { return sample_; }
    void setSeed(unsigned int seed) { seed_ = seed; }
    unsigned int getSeed() const { return seed_; }
    void filter(pcl::PointCloud<pcl::PointXYZ>&) { unavailable(); }
    void filter(std::vector<int>&) { unavailable(); }

private:
    static void unavailable() {
        throw std::runtime_error("FarthestPointSampling needs PCL 1.13+");
    }
    std::size_t sample_ = 0;
    unsigned int seed_ = 0;
#endif
};

using FPoint = pcl::PointXYZ;
using FCloud = pcl::PointCloud<FPoint>;
using FCloudPtr = std::shared_ptr<FCloud>;

inline void refineNormals(const FCloudPtr& cloud,
                          const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
                          int k, unsigned int max_iterations,
                          float convergence_threshold,
                          pcl::PointCloud<pcl::Normal>& out) {
    pcl::KdTreeFLANN<FPoint> tree;
    tree.setInputCloud(cloud);
    std::vector<pcl::Indices> k_indices(cloud->size());
    std::vector<std::vector<float>> k_sqr_distances(cloud->size());
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        tree.nearestKSearch((*cloud)[i], k, k_indices[i], k_sqr_distances[i]);
    }
    pcl::NormalRefinement<pcl::Normal> refinement(k_indices, k_sqr_distances);
    refinement.setInputCloud(normals);
    refinement.setMaxIterations(max_iterations);
    refinement.setConvergenceThreshold(convergence_threshold);
    refinement.filter(out);
}

inline void voxelGridLabel(const FCloud& cloud,
                           const std::vector<unsigned int>& labels,
                           float leaf_x, float leaf_y, float leaf_z,
                           FCloud& out_points,
                           std::vector<unsigned int>& out_labels) {
    auto labelled = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    labelled->resize(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        auto& p = (*labelled)[i];
        p.x = cloud[i].x;
        p.y = cloud[i].y;
        p.z = cloud[i].z;
        p.rgba = 0;
        p.label = labels[i];
    }
    pcl::VoxelGridLabel grid;
    grid.setInputCloud(labelled);
    grid.setLeafSize(leaf_x, leaf_y, leaf_z);
    pcl::PointCloud<pcl::PointXYZRGBL> filtered;
    grid.filter(filtered);
    out_points.resize(filtered.size());
    out_labels.resize(filtered.size());
    for (std::size_t i = 0; i < filtered.size(); ++i) {
        out_points[i] = FPoint(filtered[i].x, filtered[i].y, filtered[i].z);
        out_labels[i] = filtered[i].label;
    }
}

inline void voxelGridCovariance(const FCloudPtr& cloud, float leaf_x,
                                float leaf_y, float leaf_z, int min_points,
                                double eigenvalue_inflation,
                                std::vector<float>& out_means,
                                std::vector<float>& out_covariances,
                                std::vector<int>& out_counts) {
    pcl::VoxelGridCovariance<FPoint> grid;
    grid.setInputCloud(cloud);
    grid.setLeafSize(leaf_x, leaf_y, leaf_z);
    grid.setMinPointPerVoxel(min_points);
    grid.setCovEigValueInflationRatio(eigenvalue_inflation);
    FCloud centroids;
    grid.filter(centroids);
    out_means.clear();
    out_covariances.clear();
    out_counts.clear();
    for (const auto& entry : grid.getLeaves()) {
        const auto& leaf = entry.second;
        if (leaf.getPointCount() < grid.getMinPointPerVoxel()) {
            continue;
        }
        const Eigen::Vector3d mean = leaf.getMean();
        const Eigen::Matrix3d cov = leaf.getCov();
        for (int r = 0; r < 3; ++r) {
            out_means.push_back(static_cast<float>(mean(r)));
            for (int c = 0; c < 3; ++c) {
                out_covariances.push_back(static_cast<float>(cov(r, c)));
            }
        }
        out_counts.push_back(leaf.getPointCount());
    }
}

inline FCloudPtr withOrigin(const FCloudPtr& cloud, const float* origin3) {
    auto copy = std::make_shared<FCloud>(*cloud);
    copy->sensor_origin_ = Eigen::Vector4f(origin3[0], origin3[1], origin3[2], 0.0f);
    return copy;
}

inline void occludedVoxels(const FCloudPtr& cloud, const float* origin3,
                           float leaf_x, float leaf_y, float leaf_z,
                           std::vector<int>& out_ijk,
                           std::vector<float>& out_centers) {
    pcl::VoxelGridOcclusionEstimation<FPoint> grid;
    grid.setInputCloud(withOrigin(cloud, origin3));
    grid.setLeafSize(leaf_x, leaf_y, leaf_z);
    grid.initializeVoxelGrid();
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> occluded;
    grid.occlusionEstimationAll(occluded);
    out_ijk.clear();
    out_centers.clear();
    for (const auto& ijk : occluded) {
        const Eigen::Vector4f centre = grid.getCentroidCoordinate(ijk);
        for (int a = 0; a < 3; ++a) {
            out_ijk.push_back(ijk[a]);
            out_centers.push_back(centre[a]);
        }
    }
}

inline void occlusionStates(const FCloudPtr& cloud, const float* origin3,
                            float leaf_x, float leaf_y, float leaf_z,
                            const FCloud& queries,
                            std::vector<int>& out_states) {
    pcl::VoxelGridOcclusionEstimation<FPoint> grid;
    grid.setInputCloud(withOrigin(cloud, origin3));
    grid.setLeafSize(leaf_x, leaf_y, leaf_z);
    grid.initializeVoxelGrid();
    const Eigen::Vector3i lo = grid.getMinBoxCoordinates();
    const Eigen::Vector3i hi = grid.getMaxBoxCoordinates();
    out_states.assign(queries.size(), -1);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const Eigen::Vector3i ijk =
            grid.getGridCoordinates(queries[i].x, queries[i].y, queries[i].z);
        if ((ijk.array() < lo.array()).any() || (ijk.array() > hi.array()).any()) {
            continue;
        }
        int state = -1;
        if (grid.occlusionEstimation(state, ijk) == 0) {
            out_states[i] = state;
        }
    }
}

inline void clipPlane(const FCloud& cloud, float a, float b, float c,
                      float d, std::vector<int>& out) {
    pcl::PlaneClipper3D<FPoint> clipper(Eigen::Vector4f(a, b, c, d));
    pcl::Indices kept;
    clipper.clipPointCloud3D(cloud, kept);
    out.assign(kept.begin(), kept.end());
}

inline void clipBox(const FCloud& cloud, const float* rodrigues3,
                    const float* translation3, const float* size3,
                    std::vector<int>& out) {
    const Eigen::Vector3f rodrigues(rodrigues3[0], rodrigues3[1], rodrigues3[2]);
    const Eigen::Vector3f translation(translation3[0], translation3[1], translation3[2]);
    const Eigen::Vector3f size(size3[0], size3[1], size3[2]);
    pcl::BoxClipper3D<FPoint> clipper(rodrigues, translation, size);
    pcl::Indices kept;
    clipper.clipPointCloud3D(cloud, kept);
    out.assign(kept.begin(), kept.end());
}

template <typename Point, typename Kernel>
inline void runConvolution3D(const std::shared_ptr<pcl::PointCloud<Point>>& cloud,
                             Kernel& kernel, float sigma,
                             float threshold_relative_to_sigma, double radius,
                             unsigned int threads,
                             pcl::PointCloud<Point>& out) {
    kernel.setSigma(sigma);
    kernel.setThresholdRelativeToSigma(threshold_relative_to_sigma);
    pcl::filters::Convolution3D<Point, Point, Kernel> convolution;
    convolution.setInputCloud(cloud);
    convolution.setKernel(kernel);
    convolution.setSearchMethod(
        typename pcl::search::KdTree<Point>::Ptr(new pcl::search::KdTree<Point>()));
    convolution.setRadiusSearch(radius);
    convolution.setNumberOfThreads(threads);
    convolution.convolve(out);
}

inline void gaussianConvolution3D(const FCloudPtr& cloud, float sigma,
                                  float threshold_relative_to_sigma,
                                  double radius, unsigned int threads,
                                  FCloud& out) {
    pcl::filters::GaussianKernel<FPoint, FPoint> kernel;
    runConvolution3D(cloud, kernel, sigma, threshold_relative_to_sigma, radius,
                     threads, out);
}

inline void gaussianConvolution3DRGB(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>& cloud,
    float sigma, float threshold_relative_to_sigma, double radius,
    unsigned int threads, pcl::PointCloud<pcl::PointXYZRGB>& out) {
    pcl::filters::GaussianKernelRGB<pcl::PointXYZRGB, pcl::PointXYZRGB> kernel;
    runConvolution3D(cloud, kernel, sigma, threshold_relative_to_sigma, radius,
                     threads, out);
}

using Convolution2D = pcl::filters::Convolution<FPoint, FPoint>;
const int BORDERS_POLICY_IGNORE = Convolution2D::BORDERS_POLICY_IGNORE;
const int BORDERS_POLICY_MIRROR = Convolution2D::BORDERS_POLICY_MIRROR;
const int BORDERS_POLICY_DUPLICATE = Convolution2D::BORDERS_POLICY_DUPLICATE;

inline void convolve2D(const FCloudPtr& cloud,
                       const std::vector<float>& h_kernel,
                       const std::vector<float>& v_kernel,
                       int borders_policy, FCloud& out) {
    Convolution2D convolution;
    convolution.setBordersPolicy(borders_policy);
    convolution.setInputCloud(cloud);
    FCloud rows;
    const FCloud* current = cloud.get();
    if (!h_kernel.empty()) {
        convolution.setKernel(Eigen::Map<const Eigen::ArrayXf>(
            h_kernel.data(), static_cast<Eigen::Index>(h_kernel.size())));
        convolution.convolveRows(rows);
        current = &rows;
    }
    if (!v_kernel.empty()) {
        auto columns_in = std::make_shared<FCloud>(*current);
        convolution.setInputCloud(columns_in);
        convolution.setKernel(Eigen::Map<const Eigen::ArrayXf>(
            v_kernel.data(), static_cast<Eigen::Index>(v_kernel.size())));
        convolution.convolveCols(out);
    } else {
        out = *current;
    }
}

const int MORPH_OPEN = pcl::MORPH_OPEN;
const int MORPH_CLOSE = pcl::MORPH_CLOSE;
const int MORPH_DILATE = pcl::MORPH_DILATE;
const int MORPH_ERODE = pcl::MORPH_ERODE;

inline void morphologicalOperator(const FCloudPtr& cloud, float resolution,
                                  int morphological_operator, FCloud& out) {
    pcl::applyMorphologicalOperator<FPoint>(cloud, resolution,
                                            morphological_operator, out);
}

}  // namespace pclcompat
