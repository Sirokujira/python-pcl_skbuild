// Mirror header: self-contained stand-in for
// <pcl/surface/marching_cubes_hoppe.h>.
//
// Marching cubes over Hoppe's signed distance (distance to the nearest
// point's tangent plane), sampled on a regular grid: a mesh with its own
// vertices, like Poisson. The MarchingCubes base is flattened in;
// getGridResolution takes int& out-parameters and is left out. The
// constructor's defaulted parameters are reached through setters.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointNT>
class MarchingCubesHoppe {
public:
    MarchingCubesHoppe();

    // Inherited entry points (PCLBase / SurfaceReconstruction / MarchingCubes).
    void setInputCloud(const std::shared_ptr<PointCloud<PointNT>>& cloud);
    void reconstruct(PointCloud<PointNT>& points,
                     std::vector<Vertices>& polygons);

    void setIsoLevel(float iso_level);
    float getIsoLevel();
    void setGridResolution(int res_x, int res_y, int res_z);
    void setPercentageExtendGrid(float percentage);
    float getPercentageExtendGrid();

    void setDistanceIgnore(float dist_ignore);
    float getDistanceIgnore() const;
};

}  // namespace pcl
