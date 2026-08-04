// Histogram of Oriented Gradients (pcl::people::HOG).
//
// HOG itself is almost Cython-expressible — it takes and returns bare
// `float*`. What it does NOT tell you is how long the output buffer has
// to be, and getting that wrong is a heap overflow with no diagnostic.
// PCL's own header documents `descriptor` only as "HOG descriptor".
//
// Measured against PCL 1.14 by sentinel-filling the buffer and finding
// the written prefix, across six (h, w, bin_size, n_orients) combinations:
//
//     size = (h/bin_size - 2) * (w/bin_size - 2) * n_orients * 4
//
// so the shim owns the allocation and the caller cannot get it wrong.
//
// The rest of pcl/people is unreachable from here: person_cluster.h — and
// so HeightMap2D, HeadBasedSubclustering and
// GroundBasedPeopleDetectionApp, which all include it — does an
// unconditional `#include <pcl/visualization/pcl_visualizer.h>`, pulling
// VTK into what is otherwise pure geometry. HOG includes only
// point_types.h.
#pragma once

#include <cstddef>
#include <vector>

#include <pcl/people/hog.h>

namespace pclcompat {

/// Length of the descriptor `hogCompute` produces; <= 0 means the image
/// is too small for these settings (fewer than 3 bins on a side).
inline int hogDescriptorSize(int h, int w, int bin_size, int n_orients) {
    const int hb = h / bin_size - 2;
    const int wb = w / bin_size - 2;
    if (hb <= 0 || wb <= 0) {
        return 0;
    }
    return hb * wb * n_orients * 4;
}

/// *image* is `h * w * n_channels` floats in [0, 1], column-major as
/// PCL's own gradient code indexes it. Returns the descriptor length.
inline int hogCompute(std::vector<float>& image, int h, int w,
                      int n_channels, int bin_size, int n_orients,
                      bool soft_bin, std::vector<float>& out) {
    const int size = hogDescriptorSize(h, w, bin_size, n_orients);
    out.assign(size > 0 ? static_cast<std::size_t>(size) : 0, 0.0f);
    if (size <= 0) {
        return 0;
    }
    pcl::people::HOG hog;
    hog.compute(image.data(), h, w, n_channels, bin_size, n_orients, soft_bin,
                out.data());
    return size;
}

}  // namespace pclcompat
