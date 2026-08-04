// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/people_args.h (reached as "pcl/compat/people_args.h"
// because CMake adds src/ to the includes).
//
// The real header includes pcl/people/hog.h; this one names only what
// Cython has to see, so pxd generation still runs without PCL installed.
#pragma once

#include <vector>

namespace pclcompat {

int hogDescriptorSize(int h, int w, int bin_size, int n_orients);

int hogCompute(std::vector<float>& image, int h, int w, int n_channels,
               int bin_size, int n_orients, bool soft_bin,
               std::vector<float>& out);

}  // namespace pclcompat
