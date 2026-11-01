// Reach HarrisKeypoint3D's nested ResponseMethod enum from Cython.
//
// `setMethod` takes `HarrisKeypoint3D<PointInT, PointOutT>::ResponseMethod`
// — an enum nested inside a class TEMPLATE. Cython has no syntax for
// naming that (`Klass[A, B].MEMBER` is rejected), and C++ will not
// implicitly convert an int to it, so neither side can bridge the gap
// alone.
//
// The shim takes an int and does the cast, and re-exports the enumerators
// as namespace-scope constants so their values come from the header
// rather than being copied into Python and left to drift.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>

#include <pcl/common/intensity.h>
#include <pcl/keypoints/agast_2d.h>
#include <pcl/keypoints/harris_2d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/smoothed_surfaces_keypoint.h>
#include <pcl/keypoints/trajkovic_2d.h>
#include <pcl/keypoints/trajkovic_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace pclcompat {

using HarrisXYZI = pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;

const int HARRIS_METHOD_HARRIS = HarrisXYZI::HARRIS;
const int HARRIS_METHOD_NOBLE = HarrisXYZI::NOBLE;
const int HARRIS_METHOD_LOWE = HarrisXYZI::LOWE;
const int HARRIS_METHOD_TOMASI = HarrisXYZI::TOMASI;
const int HARRIS_METHOD_CURVATURE = HarrisXYZI::CURVATURE;

// AgastKeypoint2D's detector types.
const int AGAST_7_12S = 0;
const int AGAST_5_8 = 1;
const int OAST_9_16 = 2;

/// AgastKeypoint2D::detectKeypoints builds its detector once -- from the
/// first image's size, threshold and maximum value -- and keeps it. A
/// later image of another size is then read with the first one's row
/// length (a 120-wide frame after a 40-wide one: 0 corners where a fresh
/// detector finds 4; a smaller one reads past the end), and
/// setThreshold after the first compute does nothing. agastDetect gives
/// it a fresh detector of `kind` for every call.
inline void agastDetect(pcl::AgastKeypoint2D<pcl::PointXYZRGBA, pcl::PointUV>& detector,
                        int kind, pcl::PointCloud<pcl::PointUV>& out) {
    namespace agast = pcl::keypoints::agast;
    const auto input = detector.getInputCloud();
    const std::size_t width = input->width;
    const std::size_t height = input->height;
    const double threshold = detector.getThreshold();
    const double bmax = detector.getMaxDataValue();
    agast::AbstractAgastDetector::Ptr fresh;
    if (kind == AGAST_5_8) {
        fresh.reset(new agast::AgastDetector5_8(width, height, threshold, bmax));
    } else if (kind == OAST_9_16) {
        fresh.reset(new agast::OastDetector9_16(width, height, threshold, bmax));
    } else {
        fresh.reset(new agast::AgastDetector7_12s(width, height, threshold, bmax));
    }
    detector.setAgastDetector(fresh);
    detector.compute(out);
}

inline void setHarrisMethod(HarrisXYZI& detector, int method) {
    detector.setMethod(static_cast<HarrisXYZI::ResponseMethod>(method));
}

// Trajkovic's ComputationMethod the same way; the 2-D Harris detector's
// method goes through CorrectedHarris2D::setMethodCode below.
using Harris2DRGBA = pcl::HarrisKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>;
using Trajkovic2DRGBA = pcl::TrajkovicKeypoint2D<pcl::PointXYZRGBA, pcl::PointXYZI>;
using Trajkovic3DXYZ = pcl::TrajkovicKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;

/// pcl::HarrisKeypoint2D with detectKeypoints done over. PCL's (1.11
/// through master) has three faults, all measured in plain C++ on a
/// square in an 80 x 60 image:
/// - the derivative images are Eigen matrices `resize`d and never
///   cleared, and the left / right columns of one and the top / bottom
///   rows of the other are never written, so the border holds whatever
///   the heap held: NaN or garbage responses there. A NaN response sorts
///   first and makes the suppression threshold NaN, which rejects
///   nothing (single-threaded: 8 to 121 keypoints from one image,
///   depending on the process; 4 are right);
/// - computeSecondMomentMatrix keeps the image size in function-level
///   `static`s, so every later image is read with the first one's size;
/// - non-maximum suppression -- a greedy pass, strongest first -- runs
///   as an OpenMP loop over a shared std::vector<bool>, so threads race
///   (all cores, PCL's default: 2,866 different results in 3,000 runs).
/// This is PCL's algorithm with those fixed: every derivative written
/// (one-sided at the border, as PCL does where it does write them), the
/// image's own size, suppression in order on one thread. Window size
/// and minimal distance are private in PCL, so their setters are
/// shadowed to keep a copy.
class CorrectedHarris2D : public Harris2DRGBA {
public:
    void setMethodCode(int method) { setMethod(static_cast<ResponseMethod>(method)); }
    void setWindowWidth(int pixels) {
        window_width_ = pixels;
        Harris2DRGBA::setWindowWidth(pixels);
    }
    void setWindowHeight(int pixels) {
        window_height_ = pixels;
        Harris2DRGBA::setWindowHeight(pixels);
    }
    void setMinimalDistance(int pixels) {
        min_distance_ = pixels;
        Harris2DRGBA::setMinimalDistance(pixels);
    }

protected:
    void detectKeypoints(PointCloudOut& output) override {
        const int width = static_cast<int>(input_->width);
        const int height = static_cast<int>(input_->height);
        const std::size_t n = input_->size();
        pcl::common::IntensityFieldAccessor<pcl::PointXYZRGBA> intensity;
        auto at = [&](int u, int v) { return intensity((*input_)(u, v)); };
        // dx along the row, dy down the column; central inside, one-sided
        // at the border
        std::vector<float> dx(n), dy(n);
        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                const int ul = std::max(0, u - 1), ur = std::min(width - 1, u + 1);
                const int vu = std::max(0, v - 1), vd = std::min(height - 1, v + 1);
                dx[v * width + u] = (at(ur, v) - at(ul, v)) * 0.5f;
                dy[v * width + u] = (at(u, vd) - at(u, vu)) * 0.5f;
            }
        }
        const int hw = window_width_ / 2, hh = window_height_ / 2;
        auto& response = *response_cloud_;
        response.clear();
        response.resize(n);
        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                const std::size_t index = v * width + u;
                const pcl::PointXYZRGBA& in = (*input_)[index];
                pcl::PointXYZI& out = response[index];
                out.x = in.x;
                out.y = in.y;
                out.z = in.z;
                out.intensity = 0.f;
                if (!pcl::isXYZFinite(in)) {
                    continue;
                }
                // PCL's window: [u - hw, u + hw) x [v - hh, v + hh)
                float xx = 0.f, xy = 0.f, yy = 0.f;
                for (int wu = std::max(0, u - hw); wu < std::min(width, u + hw); ++wu) {
                    for (int wv = std::max(0, v - hh); wv < std::min(height, v + hh); ++wv) {
                        const float gx = dx[wv * width + wu], gy = dy[wv * width + wu];
                        xx += gx * gx;
                        xy += gx * gy;
                        yy += gy * gy;
                    }
                }
                const float trace = xx + yy, det = xx * yy - xy * xy;
                switch (method_) {
                    case HARRIS:
                        if (trace != 0.f) {
                            out.intensity = 0.04f + det - 0.04f * trace * trace;
                        }
                        break;
                    case NOBLE:
                        if (trace != 0.f) {
                            out.intensity = det / trace;
                        }
                        break;
                    case LOWE:
                        if (trace != 0.f) {
                            out.intensity = det / (trace * trace);
                        }
                        break;
                    case TOMASI:
                        out.intensity =
                            (xx + yy - std::sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)) / 2.0f;
                        break;
                }
            }
        }
        response.width = input_->width;
        response.height = input_->height;

        output.clear();
        if (!nonmax_) {
            output = response;
            for (std::size_t i = 0; i < n; ++i) {
                keypoints_indices_->indices.push_back(static_cast<pcl::index_t>(i));
            }
            return;
        }
        std::vector<pcl::index_t> order;
        order.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isfinite(response[i].intensity)) {
                order.push_back(static_cast<pcl::index_t>(i));
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](pcl::index_t a, pcl::index_t b) {
            return response[a].intensity > response[b].intensity;
        });
        if (order.empty()) {
            output.width = 0;
            output.height = 1;
            return;
        }
        const float threshold = threshold_ * response[order.front()].intensity;
        std::vector<char> taken(n, 0);
        for (const pcl::index_t index : order) {
            const pcl::PointXYZI& p = response[index];
            if (taken[index] || p.intensity < threshold || !pcl::isXYZFinite(p)) {
                continue;
            }
            output.push_back(p);
            keypoints_indices_->indices.push_back(index);
            const int u = index % width, v = index / width;
            for (int mu = std::max(0, u - min_distance_); mu < std::min(width, u + min_distance_); ++mu) {
                for (int mv = std::max(0, v - min_distance_); mv < std::min(height, v + min_distance_); ++mv) {
                    taken[mv * width + mu] = 1;
                }
            }
        }
        output.height = 1;
        output.width = static_cast<std::uint32_t>(output.size());
    }

private:
    // PCL's defaults
    int window_width_ = 3;
    int window_height_ = 3;
    int min_distance_ = 5;
    pcl::PointCloud<pcl::PointXYZI>::Ptr response_cloud_{new pcl::PointCloud<pcl::PointXYZI>};
};

const int TRAJKOVIC_FOUR_CORNERS = Trajkovic2DRGBA::FOUR_CORNERS;
const int TRAJKOVIC_EIGHT_CORNERS = Trajkovic2DRGBA::EIGHT_CORNERS;

inline void setTrajkovic2DMethod(Trajkovic2DRGBA& detector, int method) {
    detector.setMethod(static_cast<Trajkovic2DRGBA::ComputationMethod>(method));
}

inline void setTrajkovic3DMethod(Trajkovic3DXYZ& detector, int method) {
    detector.setMethod(static_cast<Trajkovic3DXYZ::ComputationMethod>(method));
}

// addSmoothedPointCloud takes the tree and the scale by non-const
// reference; the tree is one more object nothing above needs to see.
inline void addSmoothedCloud(
    pcl::SmoothedSurfacesKeypoint<pcl::PointXYZ, pcl::Normal>& detector,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    float scale) {
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);
    detector.addSmoothedPointCloud(cloud, normals, tree, scale);
}

}  // namespace pclcompat
