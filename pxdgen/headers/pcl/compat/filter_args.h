// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/filter_args.h.
//
// The filters whose PCL API Cython cannot state: Eigen planes, boxes and
// voxel indices, kernels held by value, enums nested in class templates,
// a point type (PointXYZRGBL) only one filter uses, and correspondences
// that have to be searched for first.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// NormalRefinement over each point's k nearest neighbours.
void refineNormals(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals, int k,
    unsigned int max_iterations, float convergence_threshold,
    pcl::PointCloud<pcl::Normal>& out);

// VoxelGridLabel: centroids plus the majority label of each voxel.
void voxelGridLabel(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                    const std::vector<unsigned int>& labels, float leaf_x,
                    float leaf_y, float leaf_z,
                    pcl::PointCloud<pcl::PointXYZ>& out_points,
                    std::vector<unsigned int>& out_labels);

// VoxelGridCovariance: mean (3), covariance (9, row-major) and point
// count of every voxel with at least `min_points` points.
void voxelGridCovariance(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    float leaf_x, float leaf_y, float leaf_z, int min_points,
    double eigenvalue_inflation, std::vector<float>& out_means,
    std::vector<float>& out_covariances, std::vector<int>& out_counts);

// VoxelGridOcclusionEstimation, seen from `origin3`: the (i, j, k) and
// centre of every occluded voxel, and the state of the voxels holding
// the `queries` (0 free, 1 occluded, -1 outside the grid).
void occludedVoxels(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const float* origin3, float leaf_x, float leaf_y, float leaf_z,
    std::vector<int>& out_ijk, std::vector<float>& out_centers);
void occlusionStates(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const float* origin3, float leaf_x, float leaf_y, float leaf_z,
    const pcl::PointCloud<pcl::PointXYZ>& queries,
    std::vector<int>& out_states);

// PlaneClipper3D: the points with a*x + b*y + c*z + d >= 0.
void clipPlane(const pcl::PointCloud<pcl::PointXYZ>& cloud, float a,
               float b, float c, float d, std::vector<int>& out);
// BoxClipper3D: the points inside a box of `size3`, rotated by the
// Rodrigues vector `rodrigues3` and centred at `translation3`.
void clipBox(const pcl::PointCloud<pcl::PointXYZ>& cloud,
             const float* rodrigues3, const float* translation3,
             const float* size3, std::vector<int>& out);

// Convolution3D with a Gaussian kernel (GaussianKernelRGB for colour).
void gaussianConvolution3D(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    float sigma, float threshold_relative_to_sigma, double radius,
    unsigned int threads, pcl::PointCloud<pcl::PointXYZ>& out);
void gaussianConvolution3DRGB(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>& cloud,
    float sigma, float threshold_relative_to_sigma, double radius,
    unsigned int threads, pcl::PointCloud<pcl::PointXYZRGB>& out);

// pcl::filters::Convolution on an organized cloud: rows with h_kernel,
// then columns with v_kernel (either may be empty to skip it).
const int BORDERS_POLICY_IGNORE = -1;
const int BORDERS_POLICY_MIRROR = 0;
const int BORDERS_POLICY_DUPLICATE = 1;
void convolve2D(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                const std::vector<float>& h_kernel,
                const std::vector<float>& v_kernel, int borders_policy,
                pcl::PointCloud<pcl::PointXYZ>& out);

// pcl::applyMorphologicalOperator on z over a grid of `resolution`.
const int MORPH_OPEN = 0;
const int MORPH_CLOSE = 1;
const int MORPH_DILATE = 2;
const int MORPH_ERODE = 3;
void morphologicalOperator(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    float resolution, int morphological_operator,
    pcl::PointCloud<pcl::PointXYZ>& out);

}  // namespace pclcompat
