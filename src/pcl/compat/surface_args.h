// Surface helpers Cython cannot state directly.
//
// - OrganizedFastMesh::TriangulationType is an enum nested in a class
//   template; Cython passes an int and this casts it.
// - BilateralUpsampling's camera matrix is an Eigen::Matrix3f. PCL never
//   initialises it (the member has no default and the constructor only
//   fills two unused Kinect presets), so the wrapper always sets one.
// - EarClipping and SimplificationRemoveUnusedVertices take and return a
//   pcl::PolygonMesh, whose vertices travel as a PCLPointCloud2 blob.
//   Meshes cross this boundary as a PointXYZ cloud plus the flat
//   (indices, counts) pair mesh_args.h uses for mesh I/O.
// - PCL 1.14's EarClipping is right on convex polygons and wrong on most
//   concave ones (measured on L, U and C shapes, both windings, jittered
//   or not: triangles covering the notch). Each polygon is clipped on its
//   own and the result checked: every triangle must face the same way as
//   the polygon (Newell normal) and their areas must sum to the
//   polygon's. A wrong answer is reported instead of returned.
#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/bilateral_upsampling.h>
#include <pcl/surface/ear_clipping.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/surface/simplification_remove_unused_vertices.h>

#include "pcl/compat/mesh_args.h"

namespace pclcompat {

using OrganizedMesh = pcl::OrganizedFastMesh<pcl::PointXYZ>;

const int ORGANIZED_MESH_TRIANGLE_RIGHT_CUT = OrganizedMesh::TRIANGLE_RIGHT_CUT;
const int ORGANIZED_MESH_TRIANGLE_LEFT_CUT = OrganizedMesh::TRIANGLE_LEFT_CUT;
const int ORGANIZED_MESH_TRIANGLE_ADAPTIVE_CUT =
    OrganizedMesh::TRIANGLE_ADAPTIVE_CUT;
const int ORGANIZED_MESH_QUAD_MESH = OrganizedMesh::QUAD_MESH;

inline void setOrganizedMeshTriangulation(OrganizedMesh& mesh, int type) {
    mesh.setTriangulationType(
        static_cast<OrganizedMesh::TriangulationType>(type));
}

inline void setBilateralProjection(
    pcl::BilateralUpsampling<pcl::PointXYZRGBA, pcl::PointXYZRGBA>& filter,
    const float* matrix9) {
    Eigen::Matrix3f m;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m(r, c) = matrix9[3 * r + c];
        }
    }
    filter.setProjectionMatrix(m);
}

inline void flattenPolygons(const std::vector<pcl::Vertices>& polygons,
                            std::vector<int>& out_indices,
                            std::vector<int>& out_counts) {
    out_indices.clear();
    out_counts.clear();
    out_counts.reserve(polygons.size());
    for (const auto& polygon : polygons) {
        out_counts.push_back(static_cast<int>(polygon.vertices.size()));
        for (const auto v : polygon.vertices) {
            out_indices.push_back(static_cast<int>(v));
        }
    }
}

/// True when `triangles` tile `polygon` exactly: all oriented like the
/// polygon's Newell normal, with areas (projected on it) that add up to
/// the polygon's own area.
inline bool tilesPolygon(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                         const pcl::Vertices& polygon,
                         const std::vector<pcl::Vertices>& triangles) {
    Eigen::Vector3d newell = Eigen::Vector3d::Zero();
    const std::size_t n = polygon.vertices.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d a =
            cloud[polygon.vertices[i]].getVector3fMap().cast<double>();
        const Eigen::Vector3d b =
            cloud[polygon.vertices[(i + 1) % n]].getVector3fMap().cast<double>();
        newell += a.cross(b);
    }
    const double area = 0.5 * newell.norm();
    if (area <= 0.0) {
        return true;  // degenerate polygon: nothing meaningful to check
    }
    const Eigen::Vector3d normal = newell.normalized();
    const double tolerance = 1e-4 * area;
    double covered = 0.0;
    for (const auto& t : triangles) {
        if (t.vertices.size() != 3) {
            return false;
        }
        const Eigen::Vector3d a = cloud[t.vertices[0]].getVector3fMap().cast<double>();
        const Eigen::Vector3d b = cloud[t.vertices[1]].getVector3fMap().cast<double>();
        const Eigen::Vector3d c = cloud[t.vertices[2]].getVector3fMap().cast<double>();
        const double projected = 0.5 * (b - a).cross(c - a).dot(normal);
        if (projected < -tolerance) {
            return false;
        }
        covered += std::abs(projected);
    }
    return std::abs(covered - area) <= tolerance;
}

inline int earClipping(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                       const std::vector<int>& indices,
                       const std::vector<int>& counts,
                       std::vector<int>& out_indices,
                       std::vector<int>& out_counts) {
    pcl::PolygonMesh all;
    buildMesh(cloud, indices, counts, all);

    std::vector<pcl::Vertices> triangles;
    for (std::size_t i = 0; i < all.polygons.size(); ++i) {
        auto single = std::make_shared<pcl::PolygonMesh>();
        single->cloud = all.cloud;
        single->polygons.push_back(all.polygons[i]);

        pcl::EarClipping clipper;
        clipper.setInputMesh(single);
        pcl::PolygonMesh output;
        clipper.process(output);
        if (!tilesPolygon(cloud, all.polygons[i], output.polygons)) {
            out_indices.clear();
            out_counts.clear();
            return static_cast<int>(i);
        }
        triangles.insert(triangles.end(), output.polygons.begin(),
                         output.polygons.end());
    }
    flattenPolygons(triangles, out_indices, out_counts);
    return -1;
}

inline void removeUnusedVertices(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                 const std::vector<int>& indices,
                                 const std::vector<int>& counts,
                                 pcl::PointCloud<pcl::PointXYZ>& out_cloud,
                                 std::vector<int>& out_indices,
                                 std::vector<int>& out_counts,
                                 std::vector<int>& out_kept) {
    pcl::PolygonMesh mesh;
    buildMesh(cloud, indices, counts, mesh);

    pcl::PolygonMesh output;
    pcl::Indices kept;
    pcl::surface::SimplificationRemoveUnusedVertices simplifier;
    simplifier.simplify(mesh, output, kept);

    pcl::fromPCLPointCloud2(output.cloud, out_cloud);
    flattenPolygons(output.polygons, out_indices, out_counts);
    out_kept.assign(kept.begin(), kept.end());
}

}  // namespace pclcompat
