// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/recognition_extra.h (reached as
// "pcl/compat/recognition_extra.h" because CMake adds src/ to the
// includes).
//
// The real header pulls in Eigen, FLANN and seven recognition headers;
// this one names only what Cython has to see.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

int crhAlign(const std::vector<float>& input_crh,
             const std::vector<float>& target_crh,
             const float* input_centroid, const float* target_centroid,
             std::vector<float>& transforms);

void trimmedICP(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& source,
                const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target,
                int num_points, float ratio, float* guess_and_result);

void occlusionFilter(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& scene,
                     const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                     float f, float threshold, bool check_invalid_depth,
                     bool occluded, pcl::PointCloud<pcl::PointXYZ>& out);

class ObjectRecognizer {
public:
    ObjectRecognizer(float pair_width, float voxel_size);
    bool addModel(const pcl::PointCloud<pcl::PointXYZ>& points,
                  const pcl::PointCloud<pcl::Normal>& normals,
                  const std::string& name);
    void setMaxCoplanarityAngleDegrees(float degrees);
    void setSceneBoundsEnlargementFactor(float factor);
    void ignoreCoplanarPointPairs(bool on);
    void icpHypothesesRefinement(bool on);
    void recognize(const pcl::PointCloud<pcl::PointXYZ>& scene,
                   const pcl::PointCloud<pcl::Normal>& normals,
                   double success_probability, std::vector<std::string>& names,
                   std::vector<float>& transforms,
                   std::vector<float>& confidences);
};

class ShapeModel {
public:
    ShapeModel();
    void setFeatureRadius(double radius);
    void setSamplingSize(float size);
    void setNumberOfClusters(unsigned int n);
    void setNVotState(bool on);
    std::size_t addTrainingCloud(const pcl::PointCloud<pcl::PointXYZ>& points,
                                 const pcl::PointCloud<pcl::Normal>& normals,
                                 unsigned int class_id);
    std::size_t numberOfTrainingClouds() const;
    bool train(const std::vector<float>& sigmas);
    unsigned int numberOfClasses() const;
    unsigned int numberOfClusters() const;
    unsigned int numberOfVisualWords() const;
    std::vector<float> sigmas() const;
    void findObjects(const pcl::PointCloud<pcl::PointXYZ>& scene,
                     const pcl::PointCloud<pcl::Normal>& normals, int class_id,
                     double non_maxima_radius, double sigma,
                     std::vector<float>& peaks);
    bool save(std::string path);
    bool load(std::string path);
};

class TemplateMatcher {
public:
    TemplateMatcher();
    int addTemplate(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& frame,
                    const std::vector<unsigned char>& mask, int x, int y,
                    int width, int height, std::size_t object_id);
    void setDetectionThreshold(float t);
    void setGradientMagnitudeThreshold(float t);
    void setIntersectionVolumeThreshold(float t);
    void detect(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& frame,
                bool semi_scale, float min_scale, float max_scale,
                float scale_multiplier, std::vector<int>& template_ids,
                std::vector<int>& object_ids, std::vector<float>& responses,
                std::vector<int>& regions, std::vector<float>& boxes);
};

class DotMatcher {
public:
    DotMatcher(std::size_t template_width, std::size_t template_height,
               std::size_t bin_size);
    void setGradientMagnitudeThreshold(float t);
    bool addTemplate(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& frame,
                     std::size_t x, std::size_t y, std::size_t& id);
    std::size_t numberOfTemplates();
    void detect(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& frame,
                float threshold, std::vector<int>& xs, std::vector<int>& ys,
                std::vector<int>& template_ids, std::vector<float>& scores);
};

}  // namespace pclcompat
