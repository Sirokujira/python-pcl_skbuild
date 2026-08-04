# distutils: language = c++
# cython: language_level=3
"""Histogram of Oriented Gradients (pcl::people::HOG).

The gradient-orientation descriptor PCL's people detector classifies:

    descriptor = pcl.hog(image, bin_size=8, n_orients=9)

*image* is a 2-D or 3-D float array with values in [0, 1].

The rest of `pcl/people` is not reachable from a build without VTK:
`person_cluster.h` does an unconditional
`#include <pcl/visualization/pcl_visualizer.h>`, and HeightMap2D,
HeadBasedSubclustering and GroundBasedPeopleDetectionApp all include it.
`GroundBasedPeopleDetectionApp` additionally needs a trained SVM file,
which is not something this package can ship. HOG includes only
point_types.h, so it stands alone.

The descriptor length is not documented by PCL; see
pcl/compat/people_args.h for the measured formula.
"""

from libcpp.vector cimport vector

from pcl.pxd.compat.people_args cimport hogCompute, hogDescriptorSize


def hog_descriptor_size(int height, int width, int bin_size=8,
                        int n_orients=9):
    """Length of the descriptor :func:`hog` returns for these settings.

    Zero means the image is too small: HOG needs at least three bins on
    each side, so ``height`` and ``width`` must each exceed
    ``2 * bin_size``.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0, got %r" % bin_size)
    if n_orients <= 0:
        raise ValueError("n_orients must be > 0, got %r" % n_orients)
    return hogDescriptorSize(height, width, bin_size, n_orients)


def hog(image, int bin_size=8, int n_orients=9, bint soft_bin=True):
    """Return the HOG descriptor of *image* as a float32 array.

    *image* is ``(height, width)`` or ``(height, width, channels)`` with
    values in [0, 1]. *bin_size* is the side of one spatial cell in
    pixels and *n_orients* the number of gradient-orientation bins;
    *soft_bin* spreads each pixel over neighbouring cells by bilinear
    interpolation, which is what PCL's own detector uses.
    """
    import numpy as np

    if bin_size <= 0:
        raise ValueError("bin_size must be > 0, got %r" % bin_size)
    if n_orients <= 0:
        raise ValueError("n_orients must be > 0, got %r" % n_orients)

    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError(
            "image must be (height, width) or (height, width, channels), "
            "got shape %r" % (values.shape,))

    cdef int height = values.shape[0]
    cdef int width = values.shape[1]
    cdef int channels = values.shape[2]

    cdef int size = hogDescriptorSize(height, width, bin_size, n_orients)
    if size <= 0:
        raise ValueError(
            "image is too small for bin_size=%d: HOG needs more than "
            "2 * bin_size pixels on each side, got %dx%d"
            % (bin_size, height, width))

    # PCL's gradient code walks the image column-major.
    flat = np.ascontiguousarray(values.transpose(2, 1, 0).reshape(-1))

    cdef vector[float] buffer
    cdef Py_ssize_t n = <Py_ssize_t> flat.shape[0]
    cdef float[::1] source = flat
    buffer.resize(n)
    cdef Py_ssize_t i
    for i in range(n):
        buffer[i] = source[i]

    cdef vector[float] out
    with nogil:
        hogCompute(buffer, height, width, channels, bin_size, n_orients,
                   soft_bin, out)

    result = np.empty(<Py_ssize_t> out.size(), dtype=np.float32)
    cdef float[::1] view = result
    for i in range(<Py_ssize_t> out.size()):
        view[i] = out[i]
    return result
