// Read and write polygon meshes.
//
// Surface reconstruction and the hulls produce triangles — and until
// this shim there was no way to persist one: a mesh could be computed
// but not saved, loaded, or handed to any other tool.
//
// `pcl::PolygonMesh` is the obstacle. It stores its points as a
// `PCLPointCloud2` (a serialized blob with a runtime field description,
// not a typed cloud) and its faces as `std::vector<pcl::Vertices>`,
// each holding its own `std::vector<uint32_t>`. Neither is something a
// pxd can usefully state, and a caller wants neither. The shim converts
// at the boundary: a typed cloud plus a flat index list in, the same
// out.
//
// Formats are the three PCL can write WITHOUT VTK. `pcl/io/vtk_lib_io.h`
// — the savePolygonFile* family — needs the VTK libraries, the same
// dependency that keeps pcl/visualization out of this package;
// `vtk_io.h` used here is PCL's own writer for the VTK ASCII format and
// needs nothing extra.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int MESH_FORMAT_PLY = 0;
const int MESH_FORMAT_OBJ = 1;
const int MESH_FORMAT_VTK = 2;

/// Build a PolygonMesh from a typed cloud and a flat polygon index list:
/// `counts[i]` indices for polygon i, taken in order from `indices`.
inline void buildMesh(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                      const std::vector<int>& indices,
                      const std::vector<int>& counts,
                      pcl::PolygonMesh& mesh) {
    pcl::toPCLPointCloud2(cloud, mesh.cloud);
    mesh.polygons.clear();
    mesh.polygons.reserve(counts.size());

    std::size_t offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        pcl::Vertices polygon;
        const std::size_t n = static_cast<std::size_t>(counts[i]);
        polygon.vertices.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            polygon.vertices.push_back(
                static_cast<std::uint32_t>(indices[offset + j]));
        }
        offset += n;
        mesh.polygons.push_back(polygon);
    }
}

/// Returns 0 on success, matching PCL's own writer convention.
inline int saveMesh(const std::string& path,
                    const pcl::PointCloud<pcl::PointXYZ>& cloud,
                    const std::vector<int>& indices,
                    const std::vector<int>& counts,
                    int format, bool binary) {
    pcl::PolygonMesh mesh;
    buildMesh(cloud, indices, counts, mesh);

    if (format == MESH_FORMAT_OBJ) {
        return pcl::io::saveOBJFile(path, mesh);
    }
    if (format == MESH_FORMAT_VTK) {
        return pcl::io::saveVTKFile(path, mesh);
    }
    if (binary) {
        return pcl::io::savePLYFileBinary(path, mesh);
    }
    return pcl::io::savePLYFile(path, mesh);
}

/// Fills the cloud and the flat polygon list. Returns 0 on success.
inline int loadMesh(const std::string& path, int format,
                    pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                    std::vector<int>& out_indices,
                    std::vector<int>& out_counts) {
    pcl::PolygonMesh mesh;
    const int error = format == MESH_FORMAT_OBJ
                          ? pcl::io::loadOBJFile(path, mesh)
                          : pcl::io::loadPLYFile(path, mesh);
    if (error != 0) {
        return error;
    }

    pcl::fromPCLPointCloud2(mesh.cloud, out_cloud);
    out_indices.clear();
    out_counts.clear();
    for (const auto& polygon : mesh.polygons) {
        out_counts.push_back(static_cast<int>(polygon.vertices.size()));
        for (const auto index : polygon.vertices) {
            out_indices.push_back(static_cast<int>(index));
        }
    }
    return 0;
}

}  // namespace pclcompat
