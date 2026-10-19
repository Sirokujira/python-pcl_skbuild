// pcl/common algorithms that numpy has no one-line equivalent for, behind
// flat arrays: PCA, the line / plane intersections, the histogram norms,
// TransformationFromCorrespondences and projection-matrix estimation.
//
// norms.h's selectNorm() handles every NormType except PF and K, which
// take two parameters; histogramNorm() calls those directly.
// estimateProjectionMatrix's `indices` defaults to {}, and the
// ConstCloudIterator it builds from that visits NO point, so called as
// documented it returns P = [1 0 0 ...] with residual 0 (1.11 through
// master); estimateProjectionMatrix() here passes every index.
#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <pcl/common/intersections.h>
#include <pcl/common/norms.h>
#include <pcl/common/pca.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

using CommonCloud = pcl::PointCloud<pcl::PointXYZ>;

/// pcl::PCA on PointXYZ: eigenvectors are columns, largest variance
/// first, the third made right-handed.
class PrincipalComponents {
public:
    void setInputCloud(const std::shared_ptr<CommonCloud>& cloud) {
        pca_.setInputCloud(cloud);
    }

    void mean(float* out3) {
        const Eigen::Vector4f m = pca_.getMean();
        for (int i = 0; i < 3; ++i) {
            out3[i] = m[i];
        }
    }

    /// Column-major 3 x 3.
    void eigenVectors(float* out9) {
        const Eigen::Matrix3f e = pca_.getEigenVectors();
        for (int i = 0; i < 9; ++i) {
            out9[i] = e.data()[i];
        }
    }

    void eigenValues(float* out3) {
        const Eigen::Vector3f v = pca_.getEigenValues();
        for (int i = 0; i < 3; ++i) {
            out3[i] = v[i];
        }
    }

    void project(const CommonCloud& in, CommonCloud& out) { pca_.project(in, out); }
    void reconstruct(const CommonCloud& in, CommonCloud& out) {
        pca_.reconstruct(in, out);
    }

private:
    pcl::PCA<pcl::PointXYZ> pca_;
};

/// pcl::lineWithLineIntersection on (point, direction) lines: the point
/// closest to both, false when they are parallel (or further apart than
/// sqrt(`sqr_eps`)).
inline bool lineWithLine(const float* a6, const float* b6, float* point3,
                         double sqr_eps) {
    Eigen::VectorXf a(6), b(6);
    for (int i = 0; i < 6; ++i) {
        a[i] = a6[i];
        b[i] = b6[i];
    }
    Eigen::Vector4f p;
    if (!pcl::lineWithLineIntersection(a, b, p, sqr_eps)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        point3[i] = p[i];
    }
    return true;
}

/// pcl::planeWithPlaneIntersection on ax + by + cz + d = 0 planes: the
/// line as (point, direction).
inline bool planeWithPlane(const float* a4, const float* b4, float* line6,
                           double angular_tolerance) {
    const Eigen::Vector4f a(a4[0], a4[1], a4[2], a4[3]);
    const Eigen::Vector4f b(b4[0], b4[1], b4[2], b4[3]);
    Eigen::VectorXf line;
    if (!pcl::planeWithPlaneIntersection(a, b, line, angular_tolerance)) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        line6[i] = line[i];
    }
    return true;
}

inline bool threePlanes(const float* a4, const float* b4, const float* c4,
                        float* point3, double determinant_tolerance) {
    const Eigen::Vector4f a(a4[0], a4[1], a4[2], a4[3]);
    const Eigen::Vector4f b(b4[0], b4[1], b4[2], b4[3]);
    const Eigen::Vector4f c(c4[0], c4[1], c4[2], c4[3]);
    Eigen::Vector3f p;
    if (!pcl::threePlanesIntersection(a, b, c, p, determinant_tolerance)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        point3[i] = p[i];
    }
    return true;
}

/// One of norms.h's distances between two equal-length vectors; `type`
/// is its NormType order. PF and K use `p1`, `p2`.
inline float histogramNorm(const std::vector<float>& a, const std::vector<float>& b,
                           int type, float p1, float p2) {
    const int dim = static_cast<int>(a.size());
    const auto norm = static_cast<pcl::NormType>(type);
    if (norm == pcl::PF) {
        return pcl::PF_Norm(a, b, dim, p1, p2);
    }
    if (norm == pcl::K) {
        return pcl::K_Norm(a, b, dim, p1, p2);
    }
    return pcl::selectNorm(a, b, dim, norm);
}

/// pcl::TransformationFromCorrespondences: the weighted least-squares
/// rigid motion taking `from` onto `to`, column-major 4 x 4.
inline void transformationFromCorrespondences(const CommonCloud& from,
                                              const CommonCloud& to,
                                              const std::vector<float>& weights,
                                              float* out16) {
    pcl::TransformationFromCorrespondences estimate;
    for (std::size_t i = 0; i < from.size(); ++i) {
        estimate.add(from[i].getVector3fMap(), to[i].getVector3fMap(),
                     weights.empty() ? 1.f : weights[i]);
    }
    const Eigen::Matrix4f m = estimate.getTransformation().matrix();
    for (int i = 0; i < 16; ++i) {
        out16[i] = m.data()[i];
    }
}

/// pcl::estimateProjectionMatrix on an organized cloud: the 3 x 4
/// projection (row-major) and its residual; `camera9` (row-major) is
/// getCameraMatrixFromProjectionMatrix's K.
inline double estimateProjectionMatrix(const std::shared_ptr<CommonCloud>& cloud,
                                       float* projection12, float* camera9) {
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> p;
    pcl::Indices all(cloud->size());
    for (std::size_t i = 0; i < all.size(); ++i) {
        all[i] = static_cast<pcl::index_t>(i);
    }
    const double residual =
        pcl::estimateProjectionMatrix<pcl::PointXYZ>(cloud, p, all);
    Eigen::Matrix3f k;
    pcl::getCameraMatrixFromProjectionMatrix(p, k);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            projection12[r * 4 + c] = p(r, c);
        }
        for (int c = 0; c < 3; ++c) {
            camera9[r * 3 + c] = k(r, c);
        }
    }
    return residual;
}

}  // namespace pclcompat
