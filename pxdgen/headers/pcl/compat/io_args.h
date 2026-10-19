// Mirror header for src/pcl/compat/io_args.h: the ASCII reader, the
// image extractors, the PCL-LZF image files and ImageGrabber read by
// index. Templates appear once per point type the wrapper uses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// IFS writer that packs x y z (PCL's own writes the padding too).
int saveIFSCloud(const std::string& path,
                 const pcl::PointCloud<pcl::PointXYZ>& cloud);

// PNG writers (VTK-backed in PCL; throw where PCL has none).
bool havePNG();
void savePNG8(const std::string& path, const std::uint8_t* data, int width,
              int height, int channels);
void savePNG16(const std::string& path, const std::uint16_t* data, int width,
               int height, int channels);

int loadASCIICloud(const std::string& path, const std::string& separators,
                   const std::string& extension,
                   pcl::PointCloud<pcl::PointXYZ>& out);

const int IMAGE_FIELD_Z = 0;
const int IMAGE_FIELD_NORMAL = 1;
const int IMAGE_FIELD_RGB = 2;
const int IMAGE_FIELD_CURVATURE = 3;
const int IMAGE_FIELD_INTENSITY = 4;
const int IMAGE_FIELD_LABEL = 5;

const int IMAGE_SCALING_NO = 0;
const int IMAGE_SCALING_FULL_RANGE = 1;
const int IMAGE_SCALING_FIXED_FACTOR = 2;

const int IMAGE_COLORS_MONO = 0;
const int IMAGE_COLORS_RGB_RANDOM = 1;
const int IMAGE_COLORS_RGB_GLASBEY = 2;

bool extractImage(const pcl::PointCloud<pcl::PointXYZ>& cloud, int field,
                  int scaling, float factor, bool paint_nans_black,
                  int color_mode, std::vector<std::uint8_t>& data,
                  int& channels, unsigned int& width, unsigned int& height);
bool extractImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, int field,
                  int scaling, float factor, bool paint_nans_black,
                  int color_mode, std::vector<std::uint8_t>& data,
                  int& channels, unsigned int& width, unsigned int& height);
bool extractImage(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud, int field,
                  int scaling, float factor, bool paint_nans_black,
                  int color_mode, std::vector<std::uint8_t>& data,
                  int& channels, unsigned int& width, unsigned int& height);
bool extractImage(const pcl::PointCloud<pcl::PointXYZI>& cloud, int field,
                  int scaling, float factor, bool paint_nans_black,
                  int color_mode, std::vector<std::uint8_t>& data,
                  int& channels, unsigned int& width, unsigned int& height);
bool extractImage(const pcl::PointCloud<pcl::Normal>& cloud, int field,
                  int scaling, float factor, bool paint_nans_black,
                  int color_mode, std::vector<std::uint8_t>& data,
                  int& channels, unsigned int& width, unsigned int& height);

bool labelImage(const std::vector<std::uint32_t>& labels, unsigned int width,
                unsigned int height, int color_mode,
                std::vector<std::uint8_t>& data, int& channels,
                unsigned int& out_width, unsigned int& out_height);

bool writePCLZFDepth(const std::string& path, const std::uint16_t* depth,
                     unsigned int width, unsigned int height);
bool writePCLZFRGB(const std::string& path, const std::uint8_t* rgb,
                   unsigned int width, unsigned int height);
bool writePCLZFParameters(const std::string& path, double fx, double fy,
                          double cx, double cy);
bool readPCLZFDepth(const std::string& path, const std::string& xml,
                    pcl::PointCloud<pcl::PointXYZ>& out);
bool readPCLZFRGBD(const std::string& depth_path, const std::string& rgb_path,
                   const std::string& xml,
                   pcl::PointCloud<pcl::PointXYZRGBA>& out);

// pcl::ImageGrabber<PointXYZRGBA>, read by index.
class ImageSequence {
public:
    ImageSequence(const std::string& dir, bool pclzf_mode);
    std::size_t size() const;
    void frame(std::size_t idx, pcl::PointCloud<pcl::PointXYZRGBA>& out) const;
    void setCameraIntrinsics(double fx, double fy, double cx, double cy);
    void cameraIntrinsics(double* out4) const;
    void setDepthImageUnits(float units);
};

}  // namespace pclcompat
