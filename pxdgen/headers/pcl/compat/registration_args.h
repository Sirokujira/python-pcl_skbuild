// Mirror header for src/pcl/compat/registration_args.h: the
// correspondence pipeline (estimation, rejection, transformation
// estimation, validation) as plain functions over flat index pairs and
// distances. pcl::Correspondences, the rejector class hierarchy and the
// Eigen matrices all stay on the C++ side.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// pairs: 2n ints (source index, target index); distances: n floats,
// SQUARED as PCL stores them. max_distance <= 0 means unlimited.
void estimateCorrespondences(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    double max_distance, bool reciprocal,
    std::vector<int>& pairs, std::vector<float>& distances);

void rejectOneToOne(const std::vector<int>& pairs,
                    const std::vector<float>& distances,
                    std::vector<int>& out_pairs,
                    std::vector<float>& out_distances);

void rejectDistance(const std::vector<int>& pairs,
                    const std::vector<float>& distances, float max_distance,
                    std::vector<int>& out_pairs,
                    std::vector<float>& out_distances);

void rejectMedianDistance(const std::vector<int>& pairs,
                          const std::vector<float>& distances, double factor,
                          std::vector<int>& out_pairs,
                          std::vector<float>& out_distances);

void rejectTrimmed(const std::vector<int>& pairs,
                   const std::vector<float>& distances, float overlap_ratio,
                   unsigned int min_correspondences,
                   std::vector<int>& out_pairs,
                   std::vector<float>& out_distances);

bool rejectSampleConsensus(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::vector<int>& pairs, const std::vector<float>& distances,
    double inlier_threshold, int max_iterations, bool refine,
    std::vector<int>& out_pairs, std::vector<float>& out_distances,
    float* out_matrix16);

void rejectSurfaceNormal(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& source_normals,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& target_normals,
    const std::vector<int>& pairs, const std::vector<float>& distances,
    double threshold, std::vector<int>& out_pairs,
    std::vector<float>& out_distances);

// method: 0 SVD, 1 Levenberg-Marquardt. An empty pairs list means point
// i of the source matches point i of the target.
bool estimateTransformation(
    int method,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::vector<int>& pairs, float* out_matrix16);

// Mean squared distance of the transformed source to the target;
// max_range <= 0 means unlimited.
double validateTransformationEuclidean(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const float* matrix16, double max_range);

}  // namespace pclcompat
