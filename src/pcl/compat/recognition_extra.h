// The rest of pcl/recognition behind Cython-expressible names:
// CRHAlignment, TrimmedICP, the organized-scene occlusion filters,
// ObjRecRANSAC, ImplicitShapeModelEstimation, LineRGBD and DOTMOD.
//
// Each one hides a type Cython cannot state (Eigen fixed-size matrices
// in aligned vectors, std::list<Output>, Histogram<153>, MaskMap /
// RegionXY by reference) and returns flat vectors instead. Several also
// absorb an upstream fault, all measured in plain C++ against 1.14:
//
// - ImplicitShapeModelEstimation casts its feature estimator to
//   FeatureFromNormals and calls through the result unchecked, so a
//   spin-image estimator (a plain Feature) segfaults; the tutorial's
//   FPFH into Histogram<153> fills 33 of the 153 values. ShapeModel uses
//   FPFH with 51 + 51 + 51 subdivisions, which fills them all.
// - Its alignYCoordWithNormal() divides by |(ny, nz)| and |(nx, ny)|, so
//   a normal exactly along x or z (any box face) gives a NaN vote, and
//   NaN normals are used as given; NaN votes then reach the kd-tree of
//   the mean-shift peak search. ShapeModel drops non-finite points and
//   tilts axis-aligned normals by 0.1 degree, as PCL already treats
//   normals near those axes.
// - Its calculateSigmas() takes the largest DOT PRODUCT between two
//   points as the object's size, so sigma depends on where the training
//   cloud sits relative to the origin. ShapeModel passes diameter / 10
//   (the stated rule of thumb), with the diameter twice the largest
//   distance from the centroid.
// - implicit_shape_model.h includes Eigen's private
//   <Eigen/src/Core/Matrix.h> before <Eigen/Core>, which only compiles
//   when something included Eigen first; this header does.
// - LineRGBD::createAndAddTemplate computes the template's 3-D box from
//   whatever cloud it is given; the modalities need the whole organized
//   frame, the box needs only the object, so TemplateMatcher hands it
//   the masked points.
// - DOTMOD measures its template size and anchor in PIXELS (the docs do
//   not say) and reads up to half a bin past the template window with
//   no bounds check; DotMatcher refuses windows that do not fit.
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/color_gradient_dot_modality.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/recognition/dotmod.h>
#include <pcl/recognition/hv/occlusion_reasoning.h>
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/linemod/line_rgbd.h>
#include <pcl/recognition/ransac_based/obj_rec_ransac.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>

// CRHAlignment calls pcl_common's kiss_fft, built with float; VTK's CMake
// target exports kiss_fft_scalar=double to its consumers, and
// src/pcl/CMakeLists.txt undoes that for every module. Measured before
// that: the same histograms gave the right roll in plain C++ and five
// wrong ones from Python.
static_assert(sizeof(kiss_fft_cpx) == 2 * sizeof(float),
              "kiss_fft_scalar must be float, as in pcl_common");

namespace pclcompat {

using RecCloud = pcl::PointCloud<pcl::PointXYZ>;
using RecNormals = pcl::PointCloud<pcl::Normal>;
using RecColourCloud = pcl::PointCloud<pcl::PointXYZRGBA>;

inline void appendColumnMajor(const Eigen::Matrix4f& m, std::vector<float>& out) {
    const float* data = m.data();
    out.insert(out.end(), data, data + 16);
}

inline Eigen::Matrix4f fromColumnMajor(const float* in16) {
    Eigen::Matrix4f m;
    float* data = m.data();
    for (int i = 0; i < 16; ++i) {
        data[i] = in16[i];
    }
    return m;
}

// --- CRHAlignment -------------------------------------------------------------

/// Candidate rigid transforms taking the input view onto the target view
/// from their CRH histograms (90 values each, as CRHEstimation returns
/// them) and centroids; column-major 4x4 each, best first. Returns how
/// many.
inline int crhAlign(const std::vector<float>& input_crh,
                    const std::vector<float>& target_crh,
                    const float* input_centroid, const float* target_centroid,
                    std::vector<float>& transforms) {
    pcl::PointCloud<pcl::Histogram<90>> input(1, 1), target(1, 1);
    for (int i = 0; i < 90; ++i) {
        input[0].histogram[i] = input_crh[i];
        target[0].histogram[i] = target_crh[i];
    }
    Eigen::Vector3f c_input(input_centroid[0], input_centroid[1],
                            input_centroid[2]);
    Eigen::Vector3f c_target(target_centroid[0], target_centroid[1],
                             target_centroid[2]);
    pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
    alignment.setInputAndTargetCentroids(c_input, c_target);
    alignment.align(input, target);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> found;
    alignment.getTransforms(found);
    transforms.clear();
    for (const auto& m : found) {
        appendColumnMajor(m, transforms);
    }
    return static_cast<int>(found.size());
}

// --- TrimmedICP ---------------------------------------------------------------

/// pcl::recognition::TrimmedICP: refine `guess_and_result` (column-major
/// 4x4, source -> target) using only the `num_points` source points
/// closest to the target each round, until an iteration lowers the
/// energy by less than 1 - `ratio`.
inline void trimmedICP(const std::shared_ptr<RecCloud>& source,
                       const std::shared_ptr<RecCloud>& target, int num_points,
                       float ratio, float* guess_and_result) {
    pcl::recognition::TrimmedICP<pcl::PointXYZ, float> icp;
    icp.init(target);
    icp.setNewToOldEnergyRatio(ratio);
    Eigen::Matrix4f m = fromColumnMajor(guess_and_result);
    icp.align(*source, num_points, m);
    const float* data = m.data();
    for (int i = 0; i < 16; ++i) {
        guess_and_result[i] = data[i];
    }
}

// --- occlusion reasoning --------------------------------------------------------

/// The points of `cloud` an organized `scene` does not hide (or, with
/// `occluded`, the ones it does), projecting with focal length `f`
/// about the image centre: pcl::occlusion_reasoning::filter and
/// getOccludedCloud.
inline void occlusionFilter(const std::shared_ptr<RecCloud>& scene,
                            const std::shared_ptr<RecCloud>& cloud, float f,
                            float threshold, bool check_invalid_depth,
                            bool occluded, RecCloud& out) {
    RecCloud::Ptr organized = scene;
    RecCloud::Ptr points = cloud;
    RecCloud::Ptr result =
        occluded ? pcl::occlusion_reasoning::getOccludedCloud<pcl::PointXYZ,
                                                               pcl::PointXYZ>(
                       organized, points, f, threshold, check_invalid_depth)
                 : pcl::occlusion_reasoning::filter<pcl::PointXYZ, pcl::PointXYZ>(
                       organized, points, f, threshold, check_invalid_depth);
    out = *result;
}

// --- ObjRecRANSAC ----------------------------------------------------------------

/// pcl::recognition::ObjRecRANSAC with its outputs flattened.
class ObjectRecognizer {
public:
    ObjectRecognizer(float pair_width, float voxel_size)
        : rec_(pair_width, voxel_size) {}

    bool addModel(const RecCloud& points, const RecNormals& normals,
                  const std::string& name) {
        return rec_.addModel(points, normals, name);
    }

    void setMaxCoplanarityAngleDegrees(float degrees) {
        rec_.setMaxCoplanarityAngleDegrees(degrees);
    }
    void setSceneBoundsEnlargementFactor(float factor) {
        rec_.setSceneBoundsEnlargementFactor(factor);
    }
    void ignoreCoplanarPointPairs(bool on) {
        if (on) {
            rec_.ignoreCoplanarPointPairsOn();
        } else {
            rec_.ignoreCoplanarPointPairsOff();
        }
    }
    void icpHypothesesRefinement(bool on) {
        if (on) {
            rec_.icpHypothesesRefinementOn();
        } else {
            rec_.icpHypothesesRefinementOff();
        }
    }

    /// One name, column-major 4x4 (model -> scene) and confidence per
    /// recognised object.
    void recognize(const RecCloud& scene, const RecNormals& normals,
                   double success_probability, std::vector<std::string>& names,
                   std::vector<float>& transforms,
                   std::vector<float>& confidences) {
        std::list<pcl::recognition::ObjRecRANSAC::Output> found;
        rec_.recognize(scene, normals, found, success_probability);
        names.clear();
        transforms.clear();
        confidences.clear();
        for (const auto& o : found) {
            // rigid_transform_ is the rotation row by row, then the
            // translation
            Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    m(r, c) = o.rigid_transform_[3 * r + c];
                }
                m(r, 3) = o.rigid_transform_[9 + r];
            }
            names.push_back(o.object_name_);
            appendColumnMajor(m, transforms);
            confidences.push_back(o.match_confidence_);
        }
    }

private:
    pcl::recognition::ObjRecRANSAC rec_;
};

// --- ImplicitShapeModelEstimation ------------------------------------------------

/// pcl::ism::ImplicitShapeModelEstimation on PointXYZ with FPFH words,
/// plus its model (see the header comment for what it corrects).
class ShapeModel {
    using Estimation =
        pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>;
    using Histogram = pcl::Histogram<153>;

public:
    ShapeModel() : model_(new pcl::features::ISMModel) {}

    void setFeatureRadius(double radius) { feature_radius_ = radius; }
    void setSamplingSize(float size) { ism_.setSamplingSize(size); }
    void setNumberOfClusters(unsigned int n) { ism_.setNumberOfClusters(n); }
    void setNVotState(bool on) { ism_.setNVotState(on); }

    /// Store one training object (non-finite points dropped); returns how
    /// many points were kept.
    std::size_t addTrainingCloud(const RecCloud& points, const RecNormals& normals,
                                 unsigned int class_id) {
        auto p = std::make_shared<RecCloud>();
        auto n = std::make_shared<RecNormals>();
        clean(points, normals, *p, *n);
        if (!p->empty()) {
            clouds_.push_back(p);
            normals_.push_back(n);
            classes_.push_back(class_id);
        }
        return p->size();
    }

    std::size_t numberOfTrainingClouds() const { return clouds_.size(); }

    /// Train; `sigmas` (one per class) empty means diameter / 10 per
    /// class, averaged over its objects.
    bool train(const std::vector<float>& sigmas) {
        if (clouds_.empty()) {
            return false;
        }
        configureEstimator();
        ism_.setTrainingClouds(clouds_);
        ism_.setTrainingNormals(normals_);
        ism_.setTrainingClasses(classes_);
        ism_.setSigmaDists(sigmas.empty() ? defaultSigmas() : sigmas);
        model_.reset(new pcl::features::ISMModel);
        return ism_.trainISM(model_);
    }

    unsigned int numberOfClasses() const { return model_->number_of_classes_; }
    unsigned int numberOfClusters() const { return model_->number_of_clusters_; }
    unsigned int numberOfVisualWords() const {
        return model_->number_of_visual_words_;
    }
    std::vector<float> sigmas() const { return model_->sigmas_; }

    /// Peaks of the votes for `class_id` as x, y, z, density, strongest
    /// first; `sigma` <= 0 takes the model's for that class.
    void findObjects(const RecCloud& scene, const RecNormals& normals,
                     int class_id, double non_maxima_radius, double sigma,
                     std::vector<float>& peaks) {
        peaks.clear();
        configureEstimator();
        auto p = std::make_shared<RecCloud>();
        auto n = std::make_shared<RecNormals>();
        clean(scene, normals, *p, *n);
        if (p->empty()) {
            return;
        }
        auto votes = ism_.findObjects(model_, p, n, class_id);
        if (votes->getNumberOfVotes() == 0) {
            return;
        }
        if (sigma <= 0) {
            sigma = model_->sigmas_[class_id];
        }
        std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak>> found;
        votes->findStrongestPeaks(found, class_id, non_maxima_radius, sigma);
        for (const auto& peak : found) {
            if (!std::isfinite(peak.x) || !std::isfinite(peak.y) ||
                !std::isfinite(peak.z) || !std::isfinite(peak.density)) {
                continue;
            }
            peaks.push_back(peak.x);
            peaks.push_back(peak.y);
            peaks.push_back(peak.z);
            peaks.push_back(static_cast<float>(peak.density));
        }
    }

    bool save(std::string path) { return model_->saveModelToFile(path); }

    /// Load a model; the search over words uses the estimator's cluster
    /// count, so it follows the file's.
    bool load(std::string path) {
        auto loaded = std::make_shared<pcl::features::ISMModel>();
        if (!loaded->loadModelFromfile(path)) {
            return false;
        }
        model_ = loaded;
        ism_.setNumberOfClusters(model_->number_of_clusters_);
        return true;
    }

private:
    void configureEstimator() {
        auto fpfh = std::make_shared<
            pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, Histogram>>();
        fpfh->setNrSubdivisions(51, 51, 51);
        fpfh->setRadiusSearch(feature_radius_);
        ism_.setFeatureEstimator(fpfh);
    }

    static void clean(const RecCloud& points, const RecNormals& normals,
                      RecCloud& p, RecNormals& n) {
        const float tilt = std::sin(0.1f * static_cast<float>(M_PI) / 180.f);
        const std::size_t count = std::min(points.size(), normals.size());
        for (std::size_t i = 0; i < count; ++i) {
            const auto& q = points[i];
            pcl::Normal m = normals[i];
            if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
                !std::isfinite(m.normal_x) || !std::isfinite(m.normal_y) ||
                !std::isfinite(m.normal_z)) {
                continue;
            }
            Eigen::Vector3f v(m.normal_x, m.normal_y, m.normal_z);
            if (v.norm() < 1e-6f) {
                continue;
            }
            v.normalize();
            if (std::hypot(v[1], v[2]) < tilt || std::hypot(v[0], v[1]) < tilt) {
                v[1] += tilt;
                v.normalize();
            }
            m.normal_x = v[0];
            m.normal_y = v[1];
            m.normal_z = v[2];
            p.push_back(q);
            n.push_back(m);
        }
    }

    std::vector<float> defaultSigmas() const {
        unsigned int count = 0;
        for (unsigned int c : classes_) {
            count = std::max(count, c + 1);
        }
        std::vector<float> sum(count, 0.f), objects(count, 0.f);
        for (std::size_t i = 0; i < clouds_.size(); ++i) {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*clouds_[i], centroid);
            float radius = 0.f;
            for (const auto& q : *clouds_[i]) {
                radius = std::max(radius,
                                  (q.getVector3fMap() - centroid.head<3>()).norm());
            }
            sum[classes_[i]] += 2.f * radius / 10.f;
            objects[classes_[i]] += 1.f;
        }
        for (unsigned int c = 0; c < count; ++c) {
            sum[c] = objects[c] > 0 ? sum[c] / objects[c] : 0.f;
        }
        return sum;
    }

    Estimation ism_;
    pcl::features::ISMModel::Ptr model_;
    double feature_radius_ = 0.05;
    std::vector<RecCloud::Ptr> clouds_;
    std::vector<RecNormals::Ptr> normals_;
    std::vector<unsigned int> classes_;
};

// --- LineRGBD ----------------------------------------------------------------------

/// pcl::LineRGBD (LINEMOD on colour gradients and surface normals) on
/// organized PointXYZRGBA frames.
class TemplateMatcher {
public:
    /// A template from an organized frame: `mask` is height * width bytes
    /// (non-zero on the object), the region its bounding rectangle.
    int addTemplate(const std::shared_ptr<RecColourCloud>& frame,
                    const std::vector<unsigned char>& mask, int x, int y,
                    int width, int height, std::size_t object_id) {
        lr_.setInputCloud(frame);
        lr_.setInputColors(frame);
        pcl::MaskMap map(frame->width, frame->height);
        RecColourCloud object;
        for (std::size_t v = 0; v < frame->height; ++v) {
            for (std::size_t u = 0; u < frame->width; ++u) {
                const bool on = mask[v * frame->width + u] != 0;
                map(u, v) = on ? 1 : 0;
                if (on) {
                    object.push_back((*frame)(u, v));
                }
            }
        }
        pcl::RegionXY region;
        region.x = x;
        region.y = y;
        region.width = width;
        region.height = height;
        return lr_.createAndAddTemplate(object, object_id, map, map, region);
    }

    void setDetectionThreshold(float t) { lr_.setDetectionThreshold(t); }
    void setGradientMagnitudeThreshold(float t) {
        lr_.setGradientMagnitudeThreshold(t);
    }
    void setIntersectionVolumeThreshold(float t) {
        lr_.setIntersectionVolumeThreshold(t);
    }

    /// Per detection: template id, object id, response, region (x, y,
    /// width, height) and 3-D box (x, y, z, width, height, depth).
    void detect(const std::shared_ptr<RecColourCloud>& frame, bool semi_scale,
                float min_scale, float max_scale, float scale_multiplier,
                std::vector<int>& template_ids, std::vector<int>& object_ids,
                std::vector<float>& responses, std::vector<int>& regions,
                std::vector<float>& boxes) {
        lr_.setInputCloud(frame);
        lr_.setInputColors(frame);
        std::vector<pcl::LineRGBD<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::Detection> found;
        if (semi_scale) {
            lr_.detectSemiScaleInvariant(found, min_scale, max_scale,
                                         scale_multiplier);
        } else {
            lr_.detect(found);
        }
        template_ids.clear();
        object_ids.clear();
        responses.clear();
        regions.clear();
        boxes.clear();
        for (const auto& d : found) {
            template_ids.push_back(static_cast<int>(d.template_id));
            object_ids.push_back(static_cast<int>(d.object_id));
            responses.push_back(d.response);
            regions.insert(regions.end(), {d.region.x, d.region.y, d.region.width,
                                           d.region.height});
            boxes.insert(boxes.end(),
                         {d.bounding_box.x, d.bounding_box.y, d.bounding_box.z,
                          d.bounding_box.width, d.bounding_box.height,
                          d.bounding_box.depth});
        }
    }

private:
    pcl::LineRGBD<pcl::PointXYZRGBA, pcl::PointXYZRGBA> lr_;
};

// --- DOTMOD --------------------------------------------------------------------------

/// pcl::DOTMOD on the colour-gradient DOT modality. Sizes, anchors and
/// detections are in pixels.
class DotMatcher {
public:
    DotMatcher(std::size_t template_width, std::size_t template_height,
               std::size_t bin_size)
        : dot_(template_width, template_height),
          modality_(bin_size),
          width_(template_width),
          height_(template_height),
          bin_(bin_size) {}

    void setGradientMagnitudeThreshold(float t) {
        modality_.setGradientMagnitudeThreshold(t);
    }

    /// A template whose window's top-left pixel is (x, y); `mask` as in
    /// TemplateMatcher. False when the window plus half a bin does not
    /// fit the frame (DOTMOD would read past it).
    bool addTemplate(const std::shared_ptr<RecColourCloud>& frame,
                     const std::vector<unsigned char>& mask, std::size_t x,
                     std::size_t y, std::size_t& id) {
        if (x + width_ + bin_ / 2 > frame->width ||
            y + height_ + bin_ / 2 > frame->height) {
            return false;
        }
        modality_.setInputCloud(frame);
        modality_.processInputData();
        pcl::MaskMap map(frame->width, frame->height);
        for (std::size_t v = 0; v < frame->height; ++v) {
            for (std::size_t u = 0; u < frame->width; ++u) {
                map(u, v) = mask[v * frame->width + u] != 0 ? 1 : 0;
            }
        }
        pcl::RegionXY region;
        region.x = static_cast<int>(x);
        region.y = static_cast<int>(y);
        region.width = static_cast<int>(width_);
        region.height = static_cast<int>(height_);
        std::vector<pcl::DOTModality*> modalities{&modality_};
        std::vector<pcl::MaskMap*> masks{&map};
        id = dot_.createAndAddTemplate(modalities, masks, x, y, region);
        return true;
    }

    std::size_t numberOfTemplates() { return dot_.getNumOfTemplates(); }

    /// Every window scoring above `threshold`: top-left pixel, template
    /// id, score (the fraction of bins whose gradients agree).
    void detect(const std::shared_ptr<RecColourCloud>& frame, float threshold,
                std::vector<int>& xs, std::vector<int>& ys,
                std::vector<int>& template_ids, std::vector<float>& scores) {
        xs.clear();
        ys.clear();
        template_ids.clear();
        scores.clear();
        if (frame->width / bin_ <= width_ / bin_ ||
            frame->height / bin_ <= height_ / bin_) {
            return;
        }
        modality_.setInputCloud(frame);
        modality_.processInputData();
        std::vector<pcl::DOTModality*> modalities{&modality_};
        std::vector<pcl::DOTMODDetection> found;
        dot_.detectTemplates(modalities, threshold, found, bin_);
        for (const auto& d : found) {
            xs.push_back(static_cast<int>(d.bin_x * bin_));
            ys.push_back(static_cast<int>(d.bin_y * bin_));
            template_ids.push_back(static_cast<int>(d.template_id));
            scores.push_back(d.score);
        }
    }

private:
    pcl::DOTMOD dot_;
    pcl::ColorGradientDOTModality<pcl::PointXYZRGBA> modality_;
    std::size_t width_;
    std::size_t height_;
    std::size_t bin_;
};

}  // namespace pclcompat
