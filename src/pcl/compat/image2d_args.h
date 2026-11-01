// pcl/2d (header-only image processing on intensity images held as
// organized PointXYZI clouds) behind flat float arrays: kernels, 2-D
// correlation, Sobel / Prewitt / Roberts / Canny / LoG edges and
// morphology.
//
// What it works around, each read in the 1.14 headers and exercised by
// tests/test_image2d.py:
//
// - Edge::detectEdgeLoG filters straight into its output with a
//   PointXYZI convolution, so it only compiles with PointXYZI output;
//   the LoG path uses Edge<PointXYZI, PointXYZI>.
// - Edge::computeDerivative{X,Y}{Central,Forward,Backward} are declared
//   but never defined (a link error if called); the same derivatives
//   are the DERIVATIVE_* kernels through imageConvolve.
// - Morphology's gray erosion / dilation use -1 as "nothing seen yet",
//   so a pixel of exactly -1 is mistaken for it; both operations commute
//   with adding a constant, so images with negative values are shifted
//   to non-negative and back.
#pragma once

#include <memory>
#include <vector>

#include <pcl/2d/convolution.h>
#include <pcl/2d/edge.h>
#include <pcl/2d/kernel.h>
#include <pcl/2d/morphology.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Image2D = pcl::PointCloud<pcl::PointXYZI>;

inline Image2D::Ptr makeImage(const float* data, int width, int height,
                              float offset = 0.f) {
    auto image = std::make_shared<Image2D>(width, height);
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            auto& p = (*image)(u, v);
            p.x = static_cast<float>(u);
            p.y = static_cast<float>(v);
            p.z = 0.f;
            p.intensity = data[v * width + u] + offset;
        }
    }
    return image;
}

inline void readImage(const Image2D& image, float* out, float offset = 0.f) {
    for (std::size_t i = 0; i < image.size(); ++i) {
        out[i] = image[i].intensity - offset;
    }
}

/// A pcl::kernel of `type` (its KERNEL_ENUM order); returns the values
/// row by row with their width and height.
inline void imageKernel(int type, int size, float sigma, std::vector<float>& values,
                        int& width, int& height) {
    pcl::kernel<pcl::PointXYZI> k;
    k.setKernelType(static_cast<pcl::kernel<pcl::PointXYZI>::KERNEL_ENUM>(type));
    k.setKernelSize(size);
    k.setKernelSigma(sigma);
    Image2D out;
    k.fetchKernel(out);
    width = static_cast<int>(out.width);
    height = static_cast<int>(out.height);
    values.resize(out.size());
    readImage(out, values.data());
}

/// pcl::Convolution (2-D): out(u, v) = sum k(l, m) in(u + l - kw/2,
/// v + m - kh/2) -- a correlation, the kernel is not flipped. `boundary`
/// is its BOUNDARY_OPTIONS_ENUM (clamp, mirror, zero padding).
inline void imageConvolve(const float* image, int width, int height,
                          const float* kernel, int kernel_width,
                          int kernel_height, int boundary, float* out) {
    pcl::Convolution<pcl::PointXYZI> convolution;
    convolution.setInputCloud(makeImage(image, width, height));
    convolution.setKernel(*makeImage(kernel, kernel_width, kernel_height));
    convolution.setBoundaryOptions(
        static_cast<pcl::Convolution<pcl::PointXYZI>::BOUNDARY_OPTIONS_ENUM>(
            boundary));
    Image2D result;
    convolution.filter(result);
    readImage(result, out);
}

/// Gradient edges: `detector` 0 Sobel, 1 Prewitt, 2 Roberts. Writes
/// magnitude, direction (atan2(gy, gx)), gx and gy.
inline void imageGradientEdges(const float* image, int width, int height,
                               int detector, float* magnitude, float* direction,
                               float* gx, float* gy) {
    pcl::Edge<pcl::PointXYZI, pcl::PointXYZIEdge> edge;
    edge.setInputCloud(makeImage(image, width, height));
    pcl::PointCloud<pcl::PointXYZIEdge> out;
    if (detector == 0) {
        edge.detectEdgeSobel(out);
    } else if (detector == 1) {
        edge.detectEdgePrewitt(out);
    } else {
        edge.detectEdgeRoberts(out);
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        magnitude[i] = out[i].magnitude;
        direction[i] = out[i].direction;
        gx[i] = out[i].magnitude_x;
        gy[i] = out[i].magnitude_y;
    }
}

/// Canny: 1 on an edge, 0 elsewhere (PCL writes 255 / 0).
inline void imageCanny(const float* image, int width, int height, float low,
                       float high, unsigned char* edges) {
    pcl::Edge<pcl::PointXYZI, pcl::PointXYZIEdge> edge;
    edge.setInputCloud(makeImage(image, width, height));
    edge.setHysteresisThresholdLow(low);
    edge.setHysteresisThresholdHigh(high);
    pcl::PointCloud<pcl::PointXYZIEdge> out;
    edge.detectEdgeCanny(out);
    for (std::size_t i = 0; i < out.size(); ++i) {
        edges[i] = out[i].magnitude > 0.f ? 1 : 0;
    }
}

/// The LoG response (Edge::detectEdgeLoG).
inline void imageLoG(const float* image, int width, int height, float sigma,
                     float size, float* out) {
    pcl::Edge<pcl::PointXYZI, pcl::PointXYZI> edge;
    edge.setInputCloud(makeImage(image, width, height));
    Image2D result;
    edge.detectEdgeLoG(sigma, size, result);
    readImage(result, out);
}

/// pcl::Morphology: `op` is its MORPHOLOGICAL_OPERATOR_TYPE (erosion,
/// dilation, opening, closing; gray then binary). `shift` moves a gray
/// image to non-negative values for the call (see the header comment).
inline void imageMorphology(const float* image, int width, int height, int op,
                            const float* element, int element_width,
                            int element_height, float shift, float* out) {
    using M = pcl::Morphology<pcl::PointXYZI>;
    M morphology;
    morphology.setInputCloud(makeImage(image, width, height, shift));
    morphology.setStructuringElement(
        makeImage(element, element_width, element_height));
    Image2D result;
    switch (static_cast<M::MORPHOLOGICAL_OPERATOR_TYPE>(op)) {
    case M::EROSION_GRAY: morphology.erosionGray(result); break;
    case M::DILATION_GRAY: morphology.dilationGray(result); break;
    case M::OPENING_GRAY: morphology.openingGray(result); break;
    case M::CLOSING_GRAY: morphology.closingGray(result); break;
    case M::EROSION_BINARY: morphology.erosionBinary(result); break;
    case M::DILATION_BINARY: morphology.dilationBinary(result); break;
    case M::OPENING_BINARY: morphology.openingBinary(result); break;
    default: morphology.closingBinary(result); break;
    }
    readImage(result, out, shift);
}

/// Binary set operations: 0 subtraction (a and not b), 1 union, 2
/// intersection.
inline void imageBinaryOperation(const float* a, const float* b, int width,
                                 int height, int op, float* out) {
    pcl::Morphology<pcl::PointXYZI> morphology;
    Image2D result;
    const auto ia = makeImage(a, width, height);
    const auto ib = makeImage(b, width, height);
    if (op == 0) {
        morphology.subtractionBinary(result, *ia, *ib);
    } else if (op == 1) {
        morphology.unionBinary(result, *ia, *ib);
    } else {
        morphology.intersectionBinary(result, *ia, *ib);
    }
    readImage(result, out);
}

/// Morphology's structuring elements: circular (radius in `a`) or
/// rectangular (`a` rows, `b` columns).
inline void structuringElement(bool circular, int a, int b,
                               std::vector<float>& values, int& width,
                               int& height) {
    pcl::Morphology<pcl::PointXYZI> morphology;
    Image2D out;
    if (circular) {
        morphology.structuringElementCircular(out, a);
    } else {
        morphology.structuringElementRectangle(out, a, b);
    }
    width = static_cast<int>(out.width);
    height = static_cast<int>(out.height);
    values.resize(out.size());
    readImage(out, values.data());
}

}  // namespace pclcompat
