// Mirror header: self-contained stand-in for
// <pcl/surface/marching_cubes_rbf.h>.
//
// Marching cubes over a radial-basis-function implicit surface fitted
// through every point and an off-surface copy of it. The fit solves a
// dense system in the number of points, so it suits small clouds. The
// MarchingCubes base is flattened in; getGridResolution takes int&
// out-parameters and is left out.
#pragma once

#include <memory>
#include <vector>

#include <pcl/Vertices.h>
#include <pcl/point_cloud.h>

namespace pcl {

template <typename PointNT>
class MarchingCubesRBF {
public:
    MarchingCubesRBF();

    // Inherited entry points (PCLBase / SurfaceReconstruction / MarchingCubes).
    void setInputCloud(const std::shared_ptr<PointCloud<PointNT>>& cloud);
    void reconstruct(PointCloud<PointNT>& points,
                     std::vector<Vertices>& polygons);

    void setIsoLevel(float iso_level);
    float getIsoLevel();
    void setGridResolution(int res_x, int res_y, int res_z);
    void setPercentageExtendGrid(float percentage);
    float getPercentageExtendGrid();

    void setOffSurfaceDisplacement(float epsilon);
    float getOffSurfaceDisplacement();
};

}  // namespace pcl
