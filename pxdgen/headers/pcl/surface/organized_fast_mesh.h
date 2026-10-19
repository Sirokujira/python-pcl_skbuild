// Mirror header: self-contained stand-in for
// <pcl/surface/organized_fast_mesh.h>.
//
// Meshes an organized cloud by connecting neighbouring pixels: no search,
// so it runs at frame rate. The polygons index the INPUT cloud, NaN
// pixels included. The nested TriangulationType enum is set through the
// pcl/compat/surface_args.h shim; setViewpoint takes an Eigen vector and
// is left at PCL's default (the origin). setMaxEdgeLength's and
// setDistanceTolerance's defaulted arguments are passed explicitly.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointInT>
class OrganizedFastMesh {
public:
    OrganizedFastMesh();

    // Inherited entry points (PCLBase / MeshConstruction).
    void setInputCloud(const std::shared_ptr<PointCloud<PointInT>>& cloud);
    void reconstruct(std::vector<Vertices>& polygons);

    void setMaxEdgeLength(float a, float b, float c);
    void unsetMaxEdgeLength();
    void setTrianglePixelSize(int triangle_size);
    void setTrianglePixelSizeRows(int triangle_size);
    void setTrianglePixelSizeColumns(int triangle_size);
    void storeShadowedFaces(bool enable);
    void setAngleTolerance(float angle_tolerance);
    void setDistanceTolerance(float distance_tolerance, bool depth_dependent);
    void useDepthAsDistance(bool enable);
};

}  // namespace pcl
