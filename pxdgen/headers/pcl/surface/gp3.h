// Mirror header: self-contained stand-in for <pcl/surface/gp3.h>
// (greedy projection triangulation).
//
// Turns a cloud with normals into a triangle mesh. `reconstruct` fills a
// vector<Vertices> — the polygons as index lists — rather than a
// PolygonMesh, which would drag in PCLPointCloud2 for no gain: the
// vertices index the input cloud the caller already has.
//
// See filters/voxel_grid.h for why the inherited methods are flattened.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT>
class GreedyProjectionTriangulation {
public:
    GreedyProjectionTriangulation();

    // Inherited entry points (PCLBase / MeshConstruction).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void reconstruct(std::vector<Vertices>& polygons);

    void setSearchRadius(double radius);
    double getSearchRadius();
    void setMu(double mu);
    double getMu();
    void setMaximumNearestNeighbors(int nnn);
    int getMaximumNearestNeighbors();
    void setMaximumSurfaceAngle(double eps_angle);
    double getMaximumSurfaceAngle();
    void setMinimumAngle(double minimum_angle);
    double getMinimumAngle();
    void setMaximumAngle(double maximum_angle);
    double getMaximumAngle();
    void setNormalConsistency(bool consistent);
    bool getNormalConsistency();
    void setConsistentVertexOrdering(bool consistent_ordering);
    bool getConsistentVertexOrdering();
};

}  // namespace pcl
