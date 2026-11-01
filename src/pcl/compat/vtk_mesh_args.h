// PCL's VTK mesh code on a PointXYZ cloud plus the flat (indices,
// counts) polygon pair mesh_args.h uses, since PolygonMesh does not
// cross into Cython: pcl/surface/vtk_smoothing -- Laplacian and
// windowed-sinc smoothing, subdivision, quadric decimation -- and
// pcl/io/vtk_lib_io.h's mesh files (STL, VTK, PLY, OBJ).
//
// Only built where PCL was built with VTK: the headers are installed
// only then, and they include VTK's.
//
// loadPolygonFile returns 0 points, not an error, for a file that is
// missing or unreadable (its vtk2mesh reports the null points and
// returns), so the wrapper checks the file first and treats 0 points as
// a failure.
//
// Two of PCL's performProcessing are replaced (1.11 through master):
// - MeshSubdivisionVTK and MeshSmoothingWindowedSincVTK hand
//   `vtk...::New()` to a vtkSmartPointer, which takes a second
//   reference, so the filter and its output mesh are never freed
//   (ASan, 900-point grid: 214 KB per subdivision, 68 KB per
//   smoothing). The overrides create them with vtkSmartPointer::New.
// - VTKUtils::vtk2mesh dereferences the filter output's points without
//   a null check, and vtkLoopSubdivisionFilter fails -- leaving them
//   null -- on any mesh with a vertex no face uses: a segfault in plain
//   C++. The subdivision override reports the failure instead.
// Laplacian smoothing and decimation neither leak nor were seen to fail
// on such meshes; processFlat still refuses an empty result.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <vtkButterflySubdivisionFilter.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkWindowedSincPolyDataFilter.h>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_quadric_decimation.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_windowed_sinc.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_subdivision.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>

#include "pcl/compat/mesh_args.h"

namespace pclcompat {

inline void flattenMesh(const pcl::PolygonMesh& mesh,
                        pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                        std::vector<int>& out_indices, std::vector<int>& out_counts) {
    out_indices.clear();
    out_counts.clear();
    pcl::fromPCLPointCloud2(mesh.cloud, out_cloud);
    for (const auto& polygon : mesh.polygons) {
        out_counts.push_back(static_cast<int>(polygon.vertices.size()));
        for (const auto index : polygon.vertices) {
            out_indices.push_back(static_cast<int>(index));
        }
    }
}

/// MeshSubdivisionVTK with its filter owned once and its failure seen.
class SubdivisionVTK : public pcl::MeshSubdivisionVTK {
public:
    /// 0 linear, 1 Loop, 2 butterfly (MeshSubdivisionVTKFilterType).
    void setFilter(int type) {
        setFilterType(static_cast<MeshSubdivisionVTKFilterType>(type));
    }

protected:
    void performProcessing(pcl::PolygonMesh& output) override {
        vtkSmartPointer<vtkPolyData> poly;
        if (pcl::VTKUtils::convertToVTK(*input_mesh_, poly) < 0) {
            output.polygons.clear();
            output.cloud.data.clear();
            return;
        }
        vtkSmartPointer<vtkPolyDataAlgorithm> filter;
        switch (getFilterType()) {
            case LOOP:
                filter = vtkSmartPointer<vtkLoopSubdivisionFilter>::New();
                break;
            case BUTTERFLY:
                filter = vtkSmartPointer<vtkButterflySubdivisionFilter>::New();
                break;
            default:
                filter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
                break;
        }
        filter->SetInputData(poly);
        filter->Update();
        poly = filter->GetOutput();
        if (poly == nullptr || poly->GetPoints() == nullptr) {
            output.polygons.clear();
            output.cloud.data.clear();
            return;
        }
        pcl::VTKUtils::convertToPCL(poly, output);
    }
};

/// MeshSmoothingWindowedSincVTK with its filter owned once.
class WindowedSincVTK : public pcl::MeshSmoothingWindowedSincVTK {
protected:
    void performProcessing(pcl::PolygonMesh& output) override {
        vtkSmartPointer<vtkPolyData> poly;
        if (pcl::VTKUtils::convertToVTK(*input_mesh_, poly) < 0) {
            output.polygons.clear();
            output.cloud.data.clear();
            return;
        }
        auto smoother = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
        smoother->SetInputData(poly);
        smoother->SetNumberOfIterations(getNumIter());
        smoother->SetPassBand(getPassBand());
        smoother->SetNormalizeCoordinates(getNormalizeCoordinates());
        smoother->SetFeatureEdgeSmoothing(getFeatureEdgeSmoothing());
        smoother->SetFeatureAngle(getFeatureAngle());
        smoother->SetEdgeAngle(getEdgeAngle());
        smoother->SetBoundarySmoothing(getBoundarySmoothing());
        smoother->Update();
        poly = smoother->GetOutput();
        if (poly == nullptr || poly->GetPoints() == nullptr) {
            output.polygons.clear();
            output.cloud.data.clear();
            return;
        }
        pcl::VTKUtils::convertToPCL(poly, output);
    }
};

/// One of the MeshProcessing classes above, called on a flat mesh.
template <typename Processor>
class FlatMeshProcessor : public Processor {
public:
    /// False when the processor returned no mesh.
    bool processFlat(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                     const std::vector<int>& indices, const std::vector<int>& counts,
                     pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                     std::vector<int>& out_indices, std::vector<int>& out_counts) {
        auto mesh = std::make_shared<pcl::PolygonMesh>();
        buildMesh(cloud, indices, counts, *mesh);
        this->setInputMesh(mesh);
        pcl::PolygonMesh output;
        this->process(output);
        out_indices.clear();
        out_counts.clear();
        if (output.cloud.data.empty()) {
            out_cloud.clear();
            return false;
        }
        flattenMesh(output, out_cloud, out_indices, out_counts);
        return true;
    }
};

/// pcl::io::loadPolygonFile: the format from the extension (.stl, .vtk,
/// .ply, .obj). The number of points read; 0 when nothing was.
inline int loadPolygonFileFlat(const std::string& path,
                               pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                               std::vector<int>& out_indices,
                               std::vector<int>& out_counts) {
    pcl::PolygonMesh mesh;
    const int n = pcl::io::loadPolygonFile(path, mesh);
    if (n <= 0 || mesh.cloud.data.empty()) {
        out_cloud.clear();
        out_indices.clear();
        out_counts.clear();
        return 0;
    }
    flattenMesh(mesh, out_cloud, out_indices, out_counts);
    return n;
}

/// pcl::io::savePolygonFile: the format from the extension (.stl, .vtk,
/// .ply).
inline bool savePolygonFileFlat(const std::string& path,
                                const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                const std::vector<int>& indices,
                                const std::vector<int>& counts, bool binary) {
    pcl::PolygonMesh mesh;
    buildMesh(cloud, indices, counts, mesh);
    return pcl::io::savePolygonFile(path, mesh, binary);
}

using LaplacianSmoothing = FlatMeshProcessor<pcl::MeshSmoothingLaplacianVTK>;
using WindowedSincSmoothing = FlatMeshProcessor<WindowedSincVTK>;
using Subdivision = FlatMeshProcessor<SubdivisionVTK>;
using QuadricDecimation = FlatMeshProcessor<pcl::MeshQuadricDecimationVTK>;

}  // namespace pclcompat
