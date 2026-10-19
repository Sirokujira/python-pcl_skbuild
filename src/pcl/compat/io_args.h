// C++ shim for the pcl/io pieces Cython cannot state: ASCIIReader's
// field setup is a member template; the image extractors are a class
// hierarchy over PCLImage with enums nested in class templates; the
// PCL-LZF image readers and writers take CameraParameters and raw
// buffers; and ImageGrabber hands out shared_ptr-to-const frames. Each
// is wrapped in one function (or one small class) with flat results.
// See pxdgen/headers/pcl/compat/io_args.h for the mirror.
//
// Absorbed upstream bugs (PCL 1.14, both still on master):
//  * saveIFSFile<PointXYZ> writes PCLPointCloud2::data verbatim, the
//    4-byte padding of every PointXYZ included, while the reader takes
//    12 bytes per vertex: every point after the first comes back shifted
//    (plain C++: the second of three points read as 1.00 1.10 1.20).
//    saveIFSCloud writes a packed x y z blob instead.
//  * ImageGrabberBase::getCloudPCLZF checks `idx > size` where it needs
//    `>=`, so the one-past-the-end index reads past the file list;
//    ImageSequence::frame bounds-checks first.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/PCLImage.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/ifs_io.h>
#include <pcl/io/image_grabber.h>
#include <pcl/io/lzf_image_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// png_io.h is installed only when PCL was built with VTK (its writers
// go through vtkPNGWriter), and PCL_BUILT_WITH_VTK is private to PCL's
// own build, so the header's presence is the test.
#if defined(__has_include)
#if __has_include(<pcl/io/png_io.h>)
#include <pcl/io/png_io.h>
#define PCLCOMPAT_HAVE_PNG 1
#endif
#endif

namespace pclcompat {

// --- PNG --------------------------------------------------------------------------

/// True when this PCL can write PNG files.
inline bool havePNG() {
#ifdef PCLCOMPAT_HAVE_PNG
    return true;
#else
    return false;
#endif
}

/// 8-bit image, `channels` interleaved per pixel (1 grey, 3 RGB, 4 RGBA).
inline void savePNG8(const std::string& path, const std::uint8_t* data,
                     int width, int height, int channels) {
#ifdef PCLCOMPAT_HAVE_PNG
    pcl::io::saveCharPNGFile(path, data, width, height, channels);
#else
    (void)path; (void)data; (void)width; (void)height; (void)channels;
    throw std::runtime_error("this PCL was built without VTK: no PNG writer");
#endif
}

/// 16-bit image (depth in millimetres, say), `channels` per pixel.
inline void savePNG16(const std::string& path, const std::uint16_t* data,
                      int width, int height, int channels) {
#ifdef PCLCOMPAT_HAVE_PNG
    pcl::io::saveShortPNGFile(path, data, width, height, channels);
#else
    (void)path; (void)data; (void)width; (void)height; (void)channels;
    throw std::runtime_error("this PCL was built without VTK: no PNG writer");
#endif
}

// --- IFS clouds --------------------------------------------------------------

/// An IFS file of the cloud's x y z, packed as the format (and PCL's own
/// reader) expects. IFS cannot hold NaN: PCL refuses a non-dense cloud.
/// 0 on success.
inline int saveIFSCloud(const std::string& path,
                        const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    pcl::PCLPointCloud2 blob;
    blob.width = static_cast<std::uint32_t>(cloud.size());
    blob.height = 1;
    blob.is_dense = cloud.is_dense;
    const char* names[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
        pcl::PCLPointField field;
        field.name = names[i];
        field.offset = 4 * i;
        field.datatype = pcl::PCLPointField::FLOAT32;
        field.count = 1;
        blob.fields.push_back(field);
    }
    blob.point_step = 12;
    blob.row_step = blob.point_step * blob.width;
    blob.data.resize(blob.row_step);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        const float xyz[3] = {cloud[i].x, cloud[i].y, cloud[i].z};
        std::memcpy(&blob.data[12 * i], xyz, sizeof(xyz));
    }
    pcl::IFSWriter writer;
    return writer.write(path, blob);
}

// --- ASCII clouds ------------------------------------------------------------

/// x y z per line, split on any of `separators`; the file must end in
/// `extension` (ASCIIReader refuses anything else). 0 on success.
inline int loadASCIICloud(const std::string& path, const std::string& separators,
                          const std::string& extension,
                          pcl::PointCloud<pcl::PointXYZ>& out) {
    pcl::ASCIIReader reader;
    reader.setInputFields<pcl::PointXYZ>();
    reader.setSepChars(separators);
    reader.setExtension(extension);
    return reader.read(path, out);
}

// --- images from organized clouds -----------------------------------------------

const int IMAGE_FIELD_Z = 0;          // mono16, scaled
const int IMAGE_FIELD_NORMAL = 1;     // rgb8
const int IMAGE_FIELD_RGB = 2;        // rgb8
const int IMAGE_FIELD_CURVATURE = 3;  // mono16, scaled
const int IMAGE_FIELD_INTENSITY = 4;  // mono16, scaled
const int IMAGE_FIELD_LABEL = 5;      // mono16 or rgb8 by colour mode

const int IMAGE_SCALING_NO =
    pcl::io::PointCloudImageExtractorWithScaling<pcl::PointXYZ>::SCALING_NO;
const int IMAGE_SCALING_FULL_RANGE = pcl::io::
    PointCloudImageExtractorWithScaling<pcl::PointXYZ>::SCALING_FULL_RANGE;
const int IMAGE_SCALING_FIXED_FACTOR = pcl::io::
    PointCloudImageExtractorWithScaling<pcl::PointXYZ>::SCALING_FIXED_FACTOR;

const int IMAGE_COLORS_MONO =
    pcl::io::PointCloudImageExtractorFromLabelField<pcl::PointXYZL>::COLORS_MONO;
const int IMAGE_COLORS_RGB_RANDOM = pcl::io::
    PointCloudImageExtractorFromLabelField<pcl::PointXYZL>::COLORS_RGB_RANDOM;
const int IMAGE_COLORS_RGB_GLASBEY = pcl::io::
    PointCloudImageExtractorFromLabelField<pcl::PointXYZL>::COLORS_RGB_GLASBEY;

/// Run one extractor; false when the cloud is not organized or lacks the
/// field. `channels` is 1 (mono16, data holds uint16 values) or 3
/// (rgb8). scaling / factor apply to Z, curvature and intensity.
template <typename PointT>
inline bool extractImage(const pcl::PointCloud<PointT>& cloud, int field,
                         int scaling, float factor, bool paint_nans_black,
                         int color_mode, std::vector<std::uint8_t>& data,
                         int& channels, unsigned int& width,
                         unsigned int& height) {
    using Scaling = typename pcl::io::PointCloudImageExtractorWithScaling<
        PointT>::ScalingMethod;
    std::shared_ptr<pcl::io::PointCloudImageExtractor<PointT>> extractor;
    switch (field) {
    case IMAGE_FIELD_Z: {
        auto e = std::make_shared<
            pcl::io::PointCloudImageExtractorFromZField<PointT>>(factor);
        e->setScalingMethod(static_cast<Scaling>(scaling));
        extractor = e;
        break;
    }
    case IMAGE_FIELD_NORMAL:
        extractor = std::make_shared<
            pcl::io::PointCloudImageExtractorFromNormalField<PointT>>();
        break;
    case IMAGE_FIELD_RGB:
        extractor = std::make_shared<
            pcl::io::PointCloudImageExtractorFromRGBField<PointT>>();
        break;
    case IMAGE_FIELD_CURVATURE: {
        auto e = std::make_shared<
            pcl::io::PointCloudImageExtractorFromCurvatureField<PointT>>();
        e->setScalingMethod(static_cast<Scaling>(scaling));
        e->setScalingFactor(factor);
        extractor = e;
        break;
    }
    case IMAGE_FIELD_INTENSITY: {
        auto e = std::make_shared<
            pcl::io::PointCloudImageExtractorFromIntensityField<PointT>>();
        e->setScalingMethod(static_cast<Scaling>(scaling));
        e->setScalingFactor(factor);
        extractor = e;
        break;
    }
    case IMAGE_FIELD_LABEL: {
        using Label = pcl::io::PointCloudImageExtractorFromLabelField<PointT>;
        extractor = std::make_shared<Label>(
            static_cast<typename Label::ColorMode>(color_mode));
        break;
    }
    default:
        throw std::invalid_argument("unknown image field");
    }
    extractor->setPaintNaNsWithBlack(paint_nans_black);
    pcl::PCLImage image;
    if (!extractor->extract(cloud, image)) {
        return false;
    }
    data.assign(image.data.begin(), image.data.end());
    channels = image.encoding == "rgb8" ? 3 : 1;
    width = image.width;
    height = image.height;
    return true;
}

/// The label extractor over a label image (one label per pixel).
inline bool labelImage(const std::vector<std::uint32_t>& labels,
                       unsigned int width, unsigned int height,
                       int color_mode, std::vector<std::uint8_t>& data,
                       int& channels, unsigned int& out_width,
                       unsigned int& out_height) {
    pcl::PointCloud<pcl::PointXYZL> cloud(width, height);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        cloud[i].x = cloud[i].y = 0.0f;
        cloud[i].z = 1.0f;
        cloud[i].label = labels[i];
    }
    return extractImage(cloud, IMAGE_FIELD_LABEL, 0, 1.0f, false, color_mode,
                        data, channels, out_width, out_height);
}

// --- PCL-LZF image files ---------------------------------------------------------

inline pcl::io::CameraParameters cameraParameters(double fx, double fy,
                                                  double cx, double cy) {
    pcl::io::CameraParameters p;
    p.focal_length_x = fx;
    p.focal_length_y = fy;
    p.principal_point_x = cx;
    p.principal_point_y = cy;
    return p;
}

/// 16-bit depth in millimetres, width x height.
inline bool writePCLZFDepth(const std::string& path,
                            const std::uint16_t* depth, unsigned int width,
                            unsigned int height) {
    pcl::io::LZFDepth16ImageWriter writer;
    return writer.write(reinterpret_cast<const char*>(depth), width, height,
                        path);
}

/// Interleaved 8-bit RGB, width x height.
inline bool writePCLZFRGB(const std::string& path, const std::uint8_t* rgb,
                          unsigned int width, unsigned int height) {
    pcl::io::LZFRGB24ImageWriter writer;
    return writer.write(reinterpret_cast<const char*>(rgb), width, height,
                        path);
}

/// The XML camera parameters, depth and RGB sections both (the writers
/// merge into an existing file).
inline bool writePCLZFParameters(const std::string& path, double fx,
                                 double fy, double cx, double cy) {
    const pcl::io::CameraParameters p = cameraParameters(fx, fy, cx, cy);
    pcl::io::LZFDepth16ImageWriter depth;
    pcl::io::LZFRGB24ImageWriter rgb;
    return depth.writeParameters(p, path) && rgb.writeParameters(p, path);
}

/// An organized cloud from a depth file and its XML parameters.
inline bool readPCLZFDepth(const std::string& path, const std::string& xml,
                           pcl::PointCloud<pcl::PointXYZ>& out) {
    pcl::io::LZFDepth16ImageReader reader;
    if (!reader.readParameters(xml)) {
        return false;
    }
    return reader.read(path, out);
}

/// An organized colour cloud from a depth file, an RGB file and their
/// XML parameters.
inline bool readPCLZFRGBD(const std::string& depth_path,
                          const std::string& rgb_path, const std::string& xml,
                          pcl::PointCloud<pcl::PointXYZRGBA>& out) {
    pcl::io::LZFRGB24ImageReader rgb;
    pcl::io::LZFDepth16ImageReader depth;
    if (!rgb.readParameters(xml) || !depth.readParameters(xml)) {
        return false;
    }
    return rgb.read(rgb_path, out) && depth.read(depth_path, out);
}

// --- ImageGrabber ---------------------------------------------------------------------

/// pcl::ImageGrabber<PointXYZRGBA> read by index rather than replayed:
/// a directory of depth (+ rgb) images -- PNG / TIFF through VTK, or
/// PCL-LZF files with one XML per frame in pclzf mode.
class ImageSequence {
public:
    ImageSequence(const std::string& dir, bool pclzf_mode)
        : grabber_(new pcl::ImageGrabber<pcl::PointXYZRGBA>(dir, 0.0f, false,
                                                             pclzf_mode)) {}

    std::size_t size() const { return grabber_->size(); }

    void frame(std::size_t idx, pcl::PointCloud<pcl::PointXYZRGBA>& out) const {
        if (idx >= grabber_->size()) {
            throw std::out_of_range("frame index out of range");
        }
        const auto cloud = (*grabber_)[idx];
        if (!cloud) {
            throw std::runtime_error("PCL could not read the frame");
        }
        out = *cloud;
    }

    void setCameraIntrinsics(double fx, double fy, double cx, double cy) {
        grabber_->setCameraIntrinsics(fx, fy, cx, cy);
    }

    void cameraIntrinsics(double* out4) const {
        grabber_->getCameraIntrinsics(out4[0], out4[1], out4[2], out4[3]);
    }

    void setDepthImageUnits(float units) {
        grabber_->setDepthImageUnits(units);
    }

private:
    std::unique_ptr<pcl::ImageGrabber<pcl::PointXYZRGBA>> grabber_;
};

}  // namespace pclcompat
