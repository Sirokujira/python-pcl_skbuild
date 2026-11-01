// Mirror header: self-contained stand-in for <pcl/surface/poisson.h>.
//
// Poisson surface reconstruction: a watertight mesh from an ORIENTED
// cloud (PointNormal). Unlike GreedyProjectionTriangulation it resamples
// the surface, so the mesh has its own vertices; the
// (points, polygons) form of reconstruct hands both back without the
// PolygonMesh / PCLPointCloud2 types.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointNT>
class Poisson {
public:
    Poisson();

    // Inherited entry points (PCLBase / SurfaceReconstruction).
    void setInputCloud(const std::shared_ptr<PointCloud<PointNT>>& cloud);
    void reconstruct(PointCloud<PointNT>& points,
                     std::vector<Vertices>& polygons);

    void setDepth(int depth);
    int getDepth();
    void setMinDepth(int min_depth);
    int getMinDepth();
    void setPointWeight(float point_weight);
    float getPointWeight();
    void setScale(float scale);
    float getScale();
    void setSolverDivide(int solver_divide);
    int getSolverDivide();
    void setIsoDivide(int iso_divide);
    int getIsoDivide();
    void setSamplesPerNode(float samples_per_node);
    float getSamplesPerNode();
    void setConfidence(bool confidence);
    bool getConfidence();
    void setOutputPolygons(bool output_polygons);
    bool getOutputPolygons();
    void setDegree(int degree);
    int getDegree();
    void setManifold(bool manifold);
    bool getManifold();
};

}  // namespace pcl
