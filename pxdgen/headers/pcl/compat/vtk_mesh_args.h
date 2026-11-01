// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/vtk_mesh_args.h (reached as
// "pcl/compat/vtk_mesh_args.h" because CMake adds src/ to the
// includes). The real header needs VTK; this one names only what
// Cython has to see. The four classes are aliases of one template over
// PCL's MeshProcessing classes, whose setters they inherit (flattened
// here).
#pragma once

#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class LaplacianSmoothing {
public:
    LaplacianSmoothing();
    void setNumIter(int num_iter);
    void setConvergence(float convergence);
    void setRelaxationFactor(float relaxation_factor);
    void setFeatureEdgeSmoothing(bool feature_edge_smoothing);
    void setFeatureAngle(float feature_angle);
    void setEdgeAngle(float edge_angle);
    void setBoundarySmoothing(bool boundary_smoothing);
    bool processFlat(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                     const std::vector<int>& indices, const std::vector<int>& counts,
                     pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                     std::vector<int>& out_indices, std::vector<int>& out_counts);
};

class WindowedSincSmoothing {
public:
    WindowedSincSmoothing();
    void setNumIter(int num_iter);
    void setPassBand(float pass_band);
    void setNormalizeCoordinates(bool normalize_coordinates);
    void setFeatureEdgeSmoothing(bool feature_edge_smoothing);
    void setFeatureAngle(float feature_angle);
    void setEdgeAngle(float edge_angle);
    void setBoundarySmoothing(bool boundary_smoothing);
    bool processFlat(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                     const std::vector<int>& indices, const std::vector<int>& counts,
                     pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                     std::vector<int>& out_indices, std::vector<int>& out_counts);
};

class Subdivision {
public:
    Subdivision();
    void setFilter(int type);
    bool processFlat(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                     const std::vector<int>& indices, const std::vector<int>& counts,
                     pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                     std::vector<int>& out_indices, std::vector<int>& out_counts);
};

class QuadricDecimation {
public:
    QuadricDecimation();
    void setTargetReductionFactor(float factor);
    bool processFlat(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                     const std::vector<int>& indices, const std::vector<int>& counts,
                     pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                     std::vector<int>& out_indices, std::vector<int>& out_counts);
};

int loadPolygonFileFlat(const std::string& path,
                        pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                        std::vector<int>& out_indices, std::vector<int>& out_counts);
bool savePolygonFileFlat(const std::string& path,
                         const pcl::PointCloud<pcl::PointXYZ>& cloud,
                         const std::vector<int>& indices,
                         const std::vector<int>& counts, bool binary);

}  // namespace pclcompat
