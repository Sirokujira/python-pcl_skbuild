// pcl::NormalBasedSignatureEstimation (Li & Guskov's normal-based
// signature) with its computeFeature corrected; see the mirror header
// for the contract.
//
// PCL's computeFeature returns only NaN (1.11 through master; 0 of
// 18,000 values finite on a sphere with exact normals, in plain C++):
//
// - the DFT over the angular samples evaluates cos(2 pi / (N k n)) --
//   the arguments inverted -- which divides by zero for every n = 0 term;
// - the DCT over the radial samples evaluates cos(pi / (M n + 0.5) k)
//   with the ROW index k, so every coefficient of a row is the same sum;
// - the sample angles are 2 pi * ((k + 1) / N) in integer division, so
//   all but the last circle sample sit at angle 0;
// - the tangent vector for a normal along z is left unnormalised.
//
// NormalSignature keeps PCL's sampling (N circles of M points, radius
// 2 (l + 1) h / M, normals averaged by inverse squared distance) and
// takes a DCT-II along each radial row and the DFT magnitude along each
// angular column, so the signature does not depend on where on the
// circle the tangent frame starts. Neighbours with non-finite normals
// are skipped; a point whose own normal is not finite gets a NaN row.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <pcl/features/normal_based_signature.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class NormalSignature
    : public pcl::NormalBasedSignatureEstimation<pcl::PointXYZ, pcl::Normal,
                                                 pcl::NormalBasedSignature12> {
    using Base = pcl::NormalBasedSignatureEstimation<pcl::PointXYZ, pcl::Normal,
                                                     pcl::NormalBasedSignature12>;

public:
    /// The signature is N' x M' = 4 x 3 values, fixed by the output type.
    static constexpr std::size_t kRows = 4;
    static constexpr std::size_t kCols = 3;

    NormalSignature() {
        setNPrime(kRows);
        setMPrime(kCols);
    }

    /// Signatures at `indices`, 12 values each (row-major N' x M'), in
    /// `out` in the same order.
    void computeAt(const std::vector<int>& indices, std::vector<float>& out) {
        setIndices(std::make_shared<pcl::Indices>(indices.begin(), indices.end()));
        FeatureCloud features;
        compute(features);
        out.clear();
        out.reserve(features.size() * kRows * kCols);
        for (const auto& f : features) {
            out.insert(out.end(), f.values, f.values + kRows * kCols);
        }
    }

protected:
    void computeFeature(FeatureCloud& output) override {
        const std::size_t n_angles = getN();
        const std::size_t n_radii = getM();
        const float h = getScale();
        if (n_angles < kRows || n_radii < kCols || !(h > 0.0f)) {
            throw std::invalid_argument(
                "NormalSignature: needs N >= 4, M >= 3 and a positive scale");
        }
        const float pi = static_cast<float>(M_PI);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        output.resize(indices_->size());

        pcl::Indices nn;
        std::vector<float> nn_sqr;
        Eigen::MatrixXf s(n_angles, n_radii);
        Eigen::MatrixXf dct(n_angles, n_radii);
        for (std::size_t index_i = 0; index_i < indices_->size(); ++index_i) {
            const auto point_i = (*indices_)[index_i];
            auto& feature = output[index_i];
            Eigen::Vector3f normal =
                (*normals_)[point_i].getNormalVector3fMap();
            if (!normal.allFinite() || normal.norm() < 1e-6f) {
                for (float& v : feature.values) {
                    v = nan;
                }
                continue;
            }
            normal.normalize();
            // PCL's tangent frame, normalised in every branch
            Eigen::Vector3f u;
            if (std::abs(normal.x()) > 0.0001f) {
                u = Eigen::Vector3f(-normal.y() / normal.x(), 1.0f, 0.0f);
            } else if (std::abs(normal.y()) > 0.0001f) {
                u = Eigen::Vector3f(1.0f, -normal.x() / normal.y(), 0.0f);
            } else {
                u = Eigen::Vector3f(0.0f, 1.0f, -normal.y() / normal.z());
            }
            u.normalize();
            const Eigen::Vector3f v = normal.cross(u);
            const Eigen::Vector3f centre =
                (*input_)[point_i].getVector3fMap();

            for (std::size_t k = 0; k < n_angles; ++k) {
                const float angle = 2.0f * pi * static_cast<float>(k + 1) /
                                    static_cast<float>(n_angles);
                const Eigen::Vector3f dir =
                    std::cos(angle) * u + std::sin(angle) * v;
                for (std::size_t l = 0; l < n_radii; ++l) {
                    const Eigen::Vector3f zeta =
                        2.0f * static_cast<float>(l + 1) * h /
                        static_cast<float>(n_radii) * dir;
                    pcl::PointXYZ query;
                    query.getVector3fMap() = centre + zeta;
                    tree_->radiusSearch(query, search_radius_, nn, nn_sqr);
                    if (nn.empty()) {
                        tree_->nearestKSearch(query, 5, nn, nn_sqr);
                    }
                    Eigen::Vector3f average = Eigen::Vector3f::Zero();
                    float weight = 0.0f;
                    for (std::size_t j = 0; j < nn.size(); ++j) {
                        const Eigen::Vector3f nj =
                            (*normals_)[nn[j]].getNormalVector3fMap();
                        if (!nj.allFinite()) {
                            continue;
                        }
                        if (nn_sqr[j] < 1e-7f) {
                            average = nj;
                            weight = 1.0f;
                            break;
                        }
                        average += nj / nn_sqr[j];
                        weight += 1.0f / nn_sqr[j];
                    }
                    s(k, l) = weight > 0.0f
                                  ? zeta.dot(average / weight) / zeta.norm()
                                  : nan;
                }
            }
            // DCT-II along each radial row
            for (std::size_t k = 0; k < n_angles; ++k) {
                for (std::size_t m = 0; m < n_radii; ++m) {
                    double sum = 0.0;
                    for (std::size_t l = 0; l < n_radii; ++l) {
                        sum += s(k, l) * std::cos(M_PI / n_radii * (l + 0.5) * m);
                    }
                    dct(k, m) = static_cast<float>(sum);
                }
            }
            // DFT magnitude along each angular column; only the kept
            // N' x M' coefficients are computed
            for (std::size_t j = 0; j < kRows; ++j) {
                for (std::size_t m = 0; m < kCols; ++m) {
                    double re = 0.0, im = 0.0;
                    for (std::size_t k = 0; k < n_angles; ++k) {
                        const double a = 2.0 * M_PI * j * k / n_angles;
                        re += dct(k, m) * std::cos(a);
                        im -= dct(k, m) * std::sin(a);
                    }
                    feature.values[j * kCols + m] =
                        static_cast<float>(std::sqrt(re * re + im * im));
                }
            }
        }
    }
};

}  // namespace pclcompat
