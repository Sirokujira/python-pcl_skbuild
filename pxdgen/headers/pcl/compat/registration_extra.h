// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/registration_extra.h.
//
// Correspondences are the flat (pairs, distances) pair of
// registration_args.h; transforms are column-major float[16]; a null
// matrix pointer means identity.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// method 2 dual quaternion, 3 SVD with scale, 4 2-D (x, y, yaw),
// 5 three-point.
bool estimateTransformationExtra(
    int method, const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::vector<int>& pairs, float* out_matrix16);

// method 0 point-to-plane (LM), 1 point-to-plane LLS, 2 point-to-plane
// LLS weighted, 3 symmetric point-to-plane LLS, 4 point-to-plane
// weighted (LM). weights: one per pair for 2 and 4.
bool estimateTransformationNormals(
    int method, const std::shared_ptr<pcl::PointCloud<pcl::PointNormal>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointNormal>>& target,
    const std::vector<int>& pairs, const std::vector<double>& weights,
    float* out_matrix16);

void correspondencesNormalShooting(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& source_normals,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target, int k,
    double max_distance, std::vector<int>& pairs, std::vector<float>& distances);
void correspondencesBackProjection(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& source_normals,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& target_normals, int k,
    double max_distance, std::vector<int>& pairs, std::vector<float>& distances);
void correspondencesOrganizedProjection(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target, float fx,
    float fy, float cx, float cy, float depth_threshold, double max_distance,
    std::vector<int>& pairs, std::vector<float>& distances);

void rejectVarTrimmed(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::vector<int>& pairs, const std::vector<float>& distances,
    double min_ratio, double max_ratio, std::vector<int>& out_pairs,
    std::vector<float>& out_distances, double* out_trimmed_distance,
    double* out_trim_factor);
void rejectPoly(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
                const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
                const std::vector<int>& pairs,
                const std::vector<float>& distances, int cardinality,
                float similarity_threshold, int iterations,
                std::vector<int>& out_pairs, std::vector<float>& out_distances);
void rejectOrganizedBoundary(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
    const std::vector<int>& pairs, const std::vector<float>& distances,
    int boundary_nans, std::vector<int>& out_pairs,
    std::vector<float>& out_distances);

bool gicp6dAlign(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& source,
                 const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& target,
                 float lab_weight, int max_iterations,
                 double max_correspondence_distance,
                 double transformation_epsilon,
                 pcl::PointCloud<pcl::PointXYZRGBA>& out, float* out_matrix16,
                 double* out_fitness);
bool ndt2dAlign(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
                const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
                const float* grid_centre2, const float* grid_extent2,
                const float* grid_step2, const double* optimisation_step3,
                int max_iterations, double transformation_epsilon,
                pcl::PointCloud<pcl::PointXYZ>& out, float* out_matrix16,
                double* out_fitness);
bool jointICPAlign(
    const std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& sources,
    const std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& targets,
    int max_iterations, double max_correspondence_distance,
    double transformation_epsilon, float* out_matrix16, double* out_fitness);
bool ppfRegister(const std::shared_ptr<pcl::PointCloud<pcl::PointNormal>>& model,
                 const std::shared_ptr<pcl::PointCloud<pcl::PointNormal>>& scene,
                 float angle_step, float distance_step,
                 unsigned int scene_sampling_rate, float position_threshold,
                 float rotation_threshold, pcl::PointCloud<pcl::PointNormal>& out,
                 float* out_matrix16, double* out_fitness);

float pyramidSimilarity(int dims, const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>& input_ranges,
                        const std::vector<float>& target_ranges);

class IncrementalICP {
public:
    IncrementalICP(double max_correspondence_distance, int max_iterations,
                   double transformation_epsilon);
    bool registerCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                       const float* delta16);
    void deltaTransform(float* out16) const;
    void absoluteTransform(float* out16) const;
    void reset();
};

class MetaICP {
public:
    MetaICP(double max_correspondence_distance, int max_iterations,
            double transformation_epsilon);
    bool registerCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                       const float* delta16);
    void absoluteTransform(float* out16) const;
    void metaCloud(pcl::PointCloud<pcl::PointXYZ>& out) const;
    void reset();
};

class LUMGraph {
public:
    LUMGraph();
    int addCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                 const float* pose6);
    void setCorrespondences(int source, int target, const std::vector<int>& pairs);
    void setMaxIterations(int count);
    void setConvergenceThreshold(float threshold);
    void compute();
    int size() const;
    void pose(int vertex, float* out6) const;
    void transformation(int vertex, float* out16) const;
    void transformedCloud(int vertex, pcl::PointCloud<pcl::PointXYZ>& out) const;
    void concatenatedCloud(pcl::PointCloud<pcl::PointXYZ>& out) const;
};

class ELCHLoop {
public:
    ELCHLoop(double max_correspondence_distance, int max_iterations,
             double transformation_epsilon);
    int addCloud(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);
    void setLoopStart(int vertex);
    void setLoopEnd(int vertex);
    void setLoopTransform(const float* matrix16);
    void compute();
    int size() const;
    void cloud(int vertex, pcl::PointCloud<pcl::PointXYZ>& out);
};

}  // namespace pclcompat
