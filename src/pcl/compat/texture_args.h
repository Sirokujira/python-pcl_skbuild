// pcl::TextureMapping::textureMeshwithMultipleCameras on a triangle mesh
// held as a cloud plus index triples, with the cameras and the result as
// flat arrays (TextureMesh's per-camera Eigen::Vector2f vectors and the
// Camera struct's Eigen::Affine3f do not cross into Cython).
#pragma once

#include <memory>
#include <vector>

#include <pcl/TextureMesh.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/texture_mapping.h>

namespace pclcompat {

/// `triangles` holds three vertex indices per face. Each camera is 22
/// values: its pose (camera -> world, column-major 4 x 4), then
/// focal_w, focal_h, center_w, center_h, width, height in pixels. Every
/// camera, in order, takes the faces it sees unoccluded among those the
/// earlier ones left; the extra last group holds the faces no camera
/// saw. Per group: its face count in `group_sizes`, the faces in
/// `faces` and three UV pairs per face in `uv` (PCL's: u right, v up,
/// in 0..1; -1 for unseen faces).
inline void textureWithCameras(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                               const std::vector<int>& triangles,
                               const std::vector<double>& cameras,
                               std::vector<int>& group_sizes, std::vector<int>& faces,
                               std::vector<float>& uv) {
    pcl::TextureMesh mesh;
    pcl::toPCLPointCloud2(*cloud, mesh.cloud);
    mesh.tex_polygons.resize(1);
    for (std::size_t i = 0; i + 2 < triangles.size(); i += 3) {
        pcl::Vertices face;
        face.vertices = {static_cast<pcl::index_t>(triangles[i]),
                         static_cast<pcl::index_t>(triangles[i + 1]),
                         static_cast<pcl::index_t>(triangles[i + 2])};
        mesh.tex_polygons[0].push_back(face);
    }
    pcl::texture_mapping::CameraVector cams;
    for (std::size_t c = 0; c + 21 < cameras.size(); c += 22) {
        pcl::texture_mapping::Camera cam;
        Eigen::Matrix4f pose;
        for (int i = 0; i < 16; ++i) {
            pose.data()[i] = static_cast<float>(cameras[c + i]);
        }
        cam.pose = Eigen::Affine3f(pose);
        cam.focal_length = cameras[c + 16];
        cam.focal_length_w = cameras[c + 16];
        cam.focal_length_h = cameras[c + 17];
        cam.center_w = cameras[c + 18];
        cam.center_h = cameras[c + 19];
        cam.width = cameras[c + 20];
        cam.height = cameras[c + 21];
        cams.push_back(cam);
    }
    pcl::TextureMapping<pcl::PointXYZ> mapping;
    mapping.textureMeshwithMultipleCameras(mesh, cams);
    group_sizes.clear();
    faces.clear();
    uv.clear();
    for (std::size_t g = 0; g < mesh.tex_polygons.size(); ++g) {
        group_sizes.push_back(static_cast<int>(mesh.tex_polygons[g].size()));
        for (const auto& face : mesh.tex_polygons[g]) {
            for (int k = 0; k < 3; ++k) {
                faces.push_back(static_cast<int>(face.vertices[k]));
            }
        }
        const auto& coords = mesh.tex_coordinates[g];
        for (std::size_t k = 0; k < 3 * mesh.tex_polygons[g].size(); ++k) {
            uv.push_back(k < coords.size() ? coords[k][0] : -1.f);
            uv.push_back(k < coords.size() ? coords[k][1] : -1.f);
        }
    }
}

}  // namespace pclcompat
