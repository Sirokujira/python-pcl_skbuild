# distutils: language = c++
# cython: language_level=3
"""pcl/2d: image processing on 2-D float arrays -- kernels, 2-D
correlation, Sobel / Prewitt / Roberts / Canny / LoG edges and
morphology.

PCL runs these on intensity images held as organized PointXYZI clouds;
here they take and return numpy arrays, (height, width) row by row. They
are PCL's implementations, kept for the organized-cloud pipelines that
use them (OrganizedEdge*, the 2-D keypoints); scipy.ndimage is the
faster general-purpose choice. Everything goes through
pcl/compat/image2d_args.h.
"""

from libcpp cimport bool as cbool
from libcpp.vector cimport vector

from pcl.pxd.compat.image2d_args cimport (
    imageBinaryOperation, imageCanny, imageConvolve, imageGradientEdges,
    imageKernel, imageLoG, imageMorphology, structuringElement)


_KERNELS = {
    "sobel_x": 0, "sobel_y": 1, "prewitt_x": 2, "prewitt_y": 3,
    "roberts_x": 4, "roberts_y": 5, "log": 6,
    "derivative_central_x": 7, "derivative_forward_x": 8,
    "derivative_backward_x": 9, "derivative_central_y": 10,
    "derivative_forward_y": 11, "derivative_backward_y": 12,
    "gaussian": 13,
}
_BOUNDARIES = {"clamp": 0, "mirror": 1, "zero": 2}
_GRADIENT_DETECTORS = {"sobel": 0, "prewitt": 1, "roberts": 2}
_MORPHOLOGY = {"erosion": 0, "dilation": 1, "opening": 2, "closing": 3}
_BINARY = {"subtraction": 0, "union": 1, "intersection": 2}


cdef _image(image, str what="image"):
    import numpy as np
    arr = np.ascontiguousarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("%s must be a non-empty (height, width) array" % what)
    return arr


cdef _from_vector(vector[float]& values, int width, int height):
    import numpy as np
    out = np.empty(height * width, dtype=np.float32)
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t> values.size()):
        out[i] = values[i]
    return out.reshape(height, width)


def image_kernel(str kind, int size=3, float sigma=1.0):
    """A pcl::kernel as a (height, width) float32 array: "sobel_x",
    "sobel_y", "prewitt_x" / "_y", "roberts_x" / "_y" (2 x 2),
    "derivative_{central,forward,backward}_{x,y}" (1 x 3 / 3 x 1),
    "gaussian" and "log" (size x size, from `size` and `sigma`).

    "gaussian" sums to 1. "log" is the Mexican hat ``(1 - r^2/2s^2)
    exp(-r^2/2s^2)`` divided by its own sum, which goes to 0 as the
    kernel outgrows sigma (sigma 1: 2.2 at size 3, 4.5e-4 at size 9), so
    its scale -- not its shape or signs -- depends on `size`."""
    if kind not in _KERNELS:
        raise ValueError("kind must be one of %s" % ", ".join(sorted(_KERNELS)))
    if kind in ("gaussian", "log"):
        if size < 1 or size % 2 == 0:
            raise ValueError("size must be odd and >= 1")
        if not sigma > 0:
            raise ValueError("sigma must be > 0")
    cdef vector[float] values
    cdef int w = 0, h = 0
    imageKernel(_KERNELS[kind], size, sigma, values, w, h)
    return _from_vector(values, w, h)


def image_convolve(image, kernel, str boundary="clamp"):
    """pcl::Convolution (2-D). It is a CORRELATION -- ``out[v, u] = sum
    k[m, l] * image[v + m - kh // 2, u + l - kw // 2]``, the kernel is not
    flipped -- with the image edge clamped, mirrored or zero-padded."""
    import numpy as np
    img = _image(image)
    k = _image(kernel, "kernel")
    if boundary not in _BOUNDARIES:
        raise ValueError("boundary must be clamp, mirror or zero")
    if boundary == "mirror" and (k.shape[0] // 2 >= img.shape[0] or
                                 k.shape[1] // 2 >= img.shape[1]):
        # PCL's mirror index goes negative past one reflection
        raise ValueError("a mirrored kernel must be smaller than the image")
    out = np.empty_like(img)
    cdef float[:, ::1] vi = img
    cdef float[:, ::1] vk = k
    cdef float[:, ::1] vo = out
    cdef int mode = _BOUNDARIES[boundary]
    with nogil:
        imageConvolve(&vi[0, 0], vi.shape[1], vi.shape[0], &vk[0, 0],
                      vk.shape[1], vk.shape[0], mode, &vo[0, 0])
    return out


def image_edges(image, str detector="sobel"):
    """Gradient edges (pcl::Edge): a dict of (height, width) float32
    arrays -- "magnitude", "direction" (atan2(gy, gx), radians), "x" and
    "y" -- for the "sobel", "prewitt" or "roberts" kernels, applied with
    `image_convolve`'s clamped correlation."""
    import numpy as np
    img = _image(image)
    if detector not in _GRADIENT_DETECTORS:
        raise ValueError("detector must be sobel, prewitt or roberts")
    magnitude = np.empty_like(img)
    direction = np.empty_like(img)
    gx = np.empty_like(img)
    gy = np.empty_like(img)
    cdef float[:, ::1] vi = img
    cdef float[:, ::1] vm = magnitude
    cdef float[:, ::1] vd = direction
    cdef float[:, ::1] vx = gx
    cdef float[:, ::1] vy = gy
    cdef int kind = _GRADIENT_DETECTORS[detector]
    with nogil:
        imageGradientEdges(&vi[0, 0], vi.shape[1], vi.shape[0], kind,
                           &vm[0, 0], &vd[0, 0], &vx[0, 0], &vy[0, 0])
    return {"magnitude": magnitude, "direction": direction, "x": gx, "y": gy}


def image_canny(image, float low=20.0, float high=80.0):
    """Canny edges (pcl::Edge::detectEdgeCanny): a (height, width) bool
    array. Gaussian 3 x 3 smoothing, Sobel gradients (unnormalised, up to
    ~1440 on an 8-bit image), non-maximum suppression and hysteresis
    between `low` and `high`. The tracing recurses once per pixel of a
    connected edge."""
    import numpy as np
    img = _image(image)
    if not 0 <= low <= high:
        raise ValueError("need 0 <= low <= high")
    edges = np.zeros(img.shape, dtype=np.uint8)
    cdef float[:, ::1] vi = img
    cdef unsigned char[:, ::1] ve = edges
    with nogil:
        imageCanny(&vi[0, 0], vi.shape[1], vi.shape[0], low, high, &ve[0, 0])
    return edges.astype(bool)


def image_log(image, float sigma=1.0, int size=3):
    """The Laplacian-of-Gaussian response (pcl::Edge::detectEdgeLoG):
    `image_convolve` with ``image_kernel("log", size, sigma)``; see that
    kernel for why only the signs and zero crossings are comparable
    between sizes."""
    import numpy as np
    img = _image(image)
    if size < 1 or size % 2 == 0 or not sigma > 0:
        raise ValueError("size must be odd and sigma > 0")
    out = np.empty_like(img)
    cdef float[:, ::1] vi = img
    cdef float[:, ::1] vo = out
    with nogil:
        imageLoG(&vi[0, 0], vi.shape[1], vi.shape[0], sigma, size, &vo[0, 0])
    return out


def structuring_element(str shape="circle", int radius=1, int height=0,
                        int width=0):
    """pcl::Morphology's structuring elements as float32 0 / 1 arrays: a
    "circle" of `radius` (on a 2 radius square, pixels strictly inside)
    or a `height` x `width` "rectangle"."""
    cdef vector[float] values
    cdef int w = 0, h = 0
    if shape == "circle":
        if radius < 1:
            raise ValueError("radius must be >= 1")
        structuringElement(True, radius, 0, values, w, h)
    elif shape == "rectangle":
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1")
        structuringElement(False, height, width, values, w, h)
    else:
        raise ValueError("shape must be circle or rectangle")
    return _from_vector(values, w, h)


def image_morphology(image, str operation, element, bint binary=False):
    """pcl::Morphology: "erosion", "dilation", "opening" or "closing" of
    `image` by `element` (a 0 / 1 array, e.g. `structuring_element`).
    Gray by default (min / max over the element); with `binary` the image
    must hold only 0 and 1. Pixels outside the image are ignored.

    PCL's gray operations take -1 to mean "nothing seen yet", so a pixel
    of exactly -1 confuses them; images with negative values are shifted
    to non-negative for the call (both operations commute with that)."""
    import numpy as np
    img = _image(image)
    se = _image(element, "element")
    if operation not in _MORPHOLOGY:
        raise ValueError("operation must be erosion, dilation, opening or "
                         "closing")
    if not np.isin(se, (0, 1)).all():
        raise ValueError("the element must hold only 0 and 1")
    cdef int op = _MORPHOLOGY[operation] + (4 if binary else 0)
    cdef float shift = 0.0
    if binary:
        if not np.isin(img, (0, 1)).all():
            raise ValueError("a binary image must hold only 0 and 1")
    else:
        lowest = float(np.nanmin(img)) if np.isfinite(img).any() else 0.0
        if lowest < 0:
            shift = -lowest
    out = np.empty_like(img)
    cdef float[:, ::1] vi = img
    cdef float[:, ::1] ve = se
    cdef float[:, ::1] vo = out
    with nogil:
        imageMorphology(&vi[0, 0], vi.shape[1], vi.shape[0], op, &ve[0, 0],
                        ve.shape[1], ve.shape[0], shift, &vo[0, 0])
    return out


def image_binary_operation(a, b, str operation):
    """pcl::Morphology's set operations on 0 / 1 images of one shape:
    "subtraction" (a and not b), "union", "intersection"."""
    import numpy as np
    ia = _image(a)
    ib = _image(b)
    if ia.shape != ib.shape:
        raise ValueError("the images must have one shape")
    if operation not in _BINARY:
        raise ValueError("operation must be subtraction, union or "
                         "intersection")
    if not (np.isin(ia, (0, 1)).all() and np.isin(ib, (0, 1)).all()):
        raise ValueError("binary images hold only 0 and 1")
    out = np.empty_like(ia)
    cdef float[:, ::1] va = ia
    cdef float[:, ::1] vb = ib
    cdef float[:, ::1] vo = out
    cdef int op = _BINARY[operation]
    with nogil:
        imageBinaryOperation(&va[0, 0], &vb[0, 0], va.shape[1], va.shape[0],
                             op, &vo[0, 0])
    return out
