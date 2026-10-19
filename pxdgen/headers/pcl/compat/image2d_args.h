// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/image2d_args.h (reached as "pcl/compat/image2d_args.h"
// because CMake adds src/ to the includes). Images are flat float
// arrays, row by row.
#pragma once

#include <vector>

namespace pclcompat {

void imageKernel(int type, int size, float sigma, std::vector<float>& values,
                 int& width, int& height);

void imageConvolve(const float* image, int width, int height,
                   const float* kernel, int kernel_width, int kernel_height,
                   int boundary, float* out);

void imageGradientEdges(const float* image, int width, int height, int detector,
                        float* magnitude, float* direction, float* gx, float* gy);

void imageCanny(const float* image, int width, int height, float low, float high,
                unsigned char* edges);

void imageLoG(const float* image, int width, int height, float sigma, float size,
              float* out);

void imageMorphology(const float* image, int width, int height, int op,
                     const float* element, int element_width, int element_height,
                     float shift, float* out);

void imageBinaryOperation(const float* a, const float* b, int width, int height,
                          int op, float* out);

void structuringElement(bool circular, int a, int b, std::vector<float>& values,
                        int& width, int& height);

}  // namespace pclcompat
