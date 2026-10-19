// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/stereo_args.h (reached as "pcl/compat/stereo_args.h"
// because CMake adds src/ to the includes).
//
// BlockStereo and AdaptiveStereo are aliases of one class template
// deriving from PCL's matchers; the setters they inherit are flattened
// onto each here.
#pragma once

#include <cstddef>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

class BlockStereo {
public:
    BlockStereo();
    void setMaxDisparity(int max_disp);
    void setXOffset(int x_off);
    void setRatioFilter(int ratio_filter);
    void setPeakFilter(int peak_filter);
    void setPreProcessing(bool is_pre_proc);
    void setLeftRightCheck(bool is_lr_check);
    void setLeftRightCheckThreshold(int lr_check_th);
    void setRadius(int radius);
    void compute(unsigned char* ref_img, unsigned char* trg_img, int width,
                 int height);
    void medianFilter(int radius);
    bool computed() const;
    int width() const;
    int height() const;
    void disparity(float* out) const;
    bool pointCloud(float u_c, float v_c, float focal, float baseline,
                    pcl::PointCloud<pcl::PointXYZ>& out);
    bool colourPointCloud(float u_c, float v_c, float focal, float baseline,
                          const unsigned char* rgb,
                          pcl::PointCloud<pcl::PointXYZRGB>& out);
};

class AdaptiveStereo {
public:
    AdaptiveStereo();
    void setMaxDisparity(int max_disp);
    void setXOffset(int x_off);
    void setRatioFilter(int ratio_filter);
    void setPeakFilter(int peak_filter);
    void setPreProcessing(bool is_pre_proc);
    void setLeftRightCheck(bool is_lr_check);
    void setLeftRightCheckThreshold(int lr_check_th);
    void setRadius(int radius);
    void setGammaS(int gamma_s);
    void setGammaC(int gamma_c);
    void setSmoothWeak(int smoothness_weak);
    void setSmoothStrong(int smoothness_strong);
    void compute(unsigned char* ref_img, unsigned char* trg_img, int width,
                 int height);
    void medianFilter(int radius);
    bool computed() const;
    int width() const;
    int height() const;
    void disparity(float* out) const;
    bool pointCloud(float u_c, float v_c, float focal, float baseline,
                    pcl::PointCloud<pcl::PointXYZ>& out);
    bool colourPointCloud(float u_c, float v_c, float focal, float baseline,
                          const unsigned char* rgb,
                          pcl::PointCloud<pcl::PointXYZRGB>& out);
};

void disparityToCloud(const std::vector<float>& disparity, std::size_t width,
                      std::size_t height, float cx, float cy, float focal,
                      float baseline, float min_disparity, float max_disparity,
                      pcl::PointCloud<pcl::PointXYZ>& out);

void disparityToIntensityCloud(const std::vector<float>& disparity,
                               std::size_t width, std::size_t height, float cx,
                               float cy, float focal, float baseline,
                               float min_disparity, float max_disparity,
                               const unsigned char* rgb,
                               pcl::PointCloud<pcl::PointXYZI>& out);

void elevationMap(std::vector<float> disparity, std::size_t width,
                  std::size_t height, float cx, float cy, float focal,
                  float baseline, float min_disparity, float max_disparity,
                  std::size_t columns, std::size_t rows, std::size_t min_points,
                  const unsigned char* rgb, std::vector<float>& cells);

}  // namespace pclcompat
