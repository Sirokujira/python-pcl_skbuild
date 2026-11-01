// pcl/stereo behind Cython-expressible names: the two dense matchers,
// DisparityMapConverter and DigitalElevationMapBuilder.
//
// The matchers keep their disparity map in a protected `short*` (16ths
// of a pixel, <= 0 for invalid); StereoMatcherT subclasses them to read
// it out as floats with NaN for invalid. The converters take a disparity
// map and an optional colour image as flat arrays.
//
// DigitalElevationMapBuilder bins by `(d - min) / ((max - min) / rows)`
// in float, and a disparity just below `max` can round to `rows` -- one
// histogram past the end. elevationMap() drops those before handing the
// map over, using the same arithmetic. StereoGrabber is not here: its
// readAhead() never reads anything (the read is commented out in 1.14
// and on master), so it publishes no frame.
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/stereo/digital_elevation_map.h>
#include <pcl/stereo/disparity_map_converter.h>
#include <pcl/stereo/stereo_matching.h>

namespace pclcompat {

template <typename Matching>
class StereoMatcherT : public Matching {
public:
    bool computed() const { return this->disp_map_ != nullptr; }
    int width() const { return this->width_; }
    int height() const { return this->height_; }

    /// The disparity of every pixel, NaN where invalid.
    void disparity(float* out) const {
        const int n = this->width_ * this->height_;
        for (int i = 0; i < n; ++i) {
            const short d = this->disp_map_[i];
            out[i] = d > 0 ? static_cast<float>(d) / 16.f
                           : std::numeric_limits<float>::quiet_NaN();
        }
    }

    bool pointCloud(float u_c, float v_c, float focal, float baseline,
                    pcl::PointCloud<pcl::PointXYZ>& out) {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        const bool ok = this->getPointCloud(u_c, v_c, focal, baseline, cloud);
        out = *cloud;
        return ok;
    }

    /// The same, coloured from `rgb` (height * width * 3 bytes).
    bool colourPointCloud(float u_c, float v_c, float focal, float baseline,
                          const unsigned char* rgb,
                          pcl::PointCloud<pcl::PointXYZRGB>& out) {
        auto texture = std::make_shared<pcl::PointCloud<pcl::RGB>>(
            this->width_, this->height_);
        for (std::size_t i = 0; i < texture->size(); ++i) {
            (*texture)[i].r = rgb[3 * i];
            (*texture)[i].g = rgb[3 * i + 1];
            (*texture)[i].b = rgb[3 * i + 2];
            (*texture)[i].a = 255;
        }
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        const bool ok =
            this->getPointCloud(u_c, v_c, focal, baseline, cloud, texture);
        out = *cloud;
        return ok;
    }
};

using BlockStereo = StereoMatcherT<pcl::BlockBasedStereoMatching>;
using AdaptiveStereo = StereoMatcherT<pcl::AdaptiveCostSOStereoMatching>;

/// pcl::DisparityMapConverter<PointXYZ>: one 3-D point per pixel whose
/// disparity lies strictly between `min_disparity` and `max_disparity`,
/// NaN elsewhere.
inline void disparityToCloud(const std::vector<float>& disparity,
                             std::size_t width, std::size_t height, float cx,
                             float cy, float focal, float baseline,
                             float min_disparity, float max_disparity,
                             pcl::PointCloud<pcl::PointXYZ>& out) {
    pcl::DisparityMapConverter<pcl::PointXYZ> converter;
    converter.setImageCenterX(cx);
    converter.setImageCenterY(cy);
    converter.setFocalLength(focal);
    converter.setBaseline(baseline);
    converter.setDisparityThresholdMin(min_disparity);
    converter.setDisparityThresholdMax(max_disparity);
    converter.setDisparityMap(disparity, width, height);
    converter.compute(out);
}

inline std::shared_ptr<pcl::PointCloud<pcl::RGB>> rgbImage(
    const unsigned char* rgb, std::size_t width, std::size_t height) {
    auto image = std::make_shared<pcl::PointCloud<pcl::RGB>>(width, height);
    for (std::size_t i = 0; i < width * height; ++i) {
        (*image)[i].r = rgb[3 * i];
        (*image)[i].g = rgb[3 * i + 1];
        (*image)[i].b = rgb[3 * i + 2];
        (*image)[i].a = 255;
    }
    return image;
}

/// The same with each point's intensity, the mean of the pixel's r, g, b
/// (DisparityMapConverter<PointXYZI> with an image).
inline void disparityToIntensityCloud(const std::vector<float>& disparity,
                                      std::size_t width, std::size_t height,
                                      float cx, float cy, float focal,
                                      float baseline, float min_disparity,
                                      float max_disparity,
                                      const unsigned char* rgb,
                                      pcl::PointCloud<pcl::PointXYZI>& out) {
    pcl::DisparityMapConverter<pcl::PointXYZI> converter;
    converter.setImageCenterX(cx);
    converter.setImageCenterY(cy);
    converter.setFocalLength(focal);
    converter.setBaseline(baseline);
    converter.setDisparityThresholdMin(min_disparity);
    converter.setDisparityThresholdMax(max_disparity);
    converter.setImage(rgbImage(rgb, width, height));
    converter.setDisparityMap(disparity, width, height);
    converter.compute(out);
}

/// pcl::DigitalElevationMapBuilder: `columns` x `rows` cells of x, y
/// (mean height), z, intensity, intensity variance, height variance,
/// row by row (rows are disparity bins from `min_disparity`).
inline void elevationMap(std::vector<float> disparity, std::size_t width,
                         std::size_t height, float cx, float cy, float focal,
                         float baseline, float min_disparity,
                         float max_disparity, std::size_t columns,
                         std::size_t rows, std::size_t min_points,
                         const unsigned char* rgb, std::vector<float>& cells) {
    const float step = (max_disparity - min_disparity) / rows;
    for (float& d : disparity) {
        if (min_disparity < d && d < max_disparity &&
            static_cast<std::size_t>((d - min_disparity) / step) >= rows) {
            d = std::numeric_limits<float>::quiet_NaN();
        }
    }
    pcl::DigitalElevationMapBuilder builder;
    builder.setImageCenterX(cx);
    builder.setImageCenterY(cy);
    builder.setFocalLength(focal);
    builder.setBaseline(baseline);
    builder.setDisparityThresholdMin(min_disparity);
    builder.setDisparityThresholdMax(max_disparity);
    builder.setResolution(columns, rows);
    builder.setMinPointsInCell(min_points);
    builder.setImage(rgbImage(rgb, width, height));
    builder.setDisparityMap(disparity, width, height);
    pcl::PointCloud<pcl::PointDEM> dem;
    builder.compute(dem);
    cells.clear();
    cells.reserve(dem.size() * 6);
    for (const auto& p : dem) {
        cells.insert(cells.end(), {p.x, p.y, p.z, p.intensity,
                                   p.intensity_variance, p.height_variance});
    }
}

}  // namespace pclcompat
