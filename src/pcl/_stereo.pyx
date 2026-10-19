# distutils: language = c++
# cython: language_level=3
"""Dense stereo (pcl/stereo): disparity from a rectified grey image
pair, and from a disparity map to 3-D points or a digital elevation map.

    bm = pcl.BlockBasedStereoMatching(max_disparity=64, radius=5)
    bm.compute(left, right)                  # (h, w) uint8 each
    disparity = bm.get_DisparityMap()        # float32, NaN where invalid
    cloud = bm.get_PointCloud(u_c, v_c, focal, baseline)

The matchers keep the map in 16ths of a pixel; it comes back in
pixels. Everything goes through pcl/compat/stereo_args.h.
StereoGrabber is not wrapped: PCL's never reads a frame (see the
README).
"""

from cython.operator cimport dereference as deref
from libcpp cimport bool as cbool
from libcpp.vector cimport vector

from pcl.pxd.compat.stereo_args cimport (
    AdaptiveStereo, BlockStereo, disparityToCloud, disparityToIntensityCloud,
    elevationMap)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.point_types cimport PointXYZ, PointXYZI, PointXYZRGB

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_PointXYZI, PointCloud_PointXYZRGB


ctypedef fused _Matcher:
    BlockStereo
    AdaptiveStereo


cdef _grey_pair(left, right):
    import numpy as np
    a = np.ascontiguousarray(left)
    b = np.ascontiguousarray(right)
    if a.ndim != 2 or a.dtype != np.uint8 or b.shape != a.shape or \
            b.dtype != np.uint8:
        raise ValueError("expected two (h, w) uint8 grey images of one size")
    return a, b


cdef _configure(_Matcher* m, int max_disparity, int x_offset, int ratio_filter,
                int peak_filter, bint pre_processing, bint left_right_check,
                int left_right_check_threshold):
    if max_disparity < 1:
        raise ValueError("max_disparity must be >= 1")
    m.setMaxDisparity(max_disparity)
    m.setXOffset(x_offset)
    m.setRatioFilter(ratio_filter)
    m.setPeakFilter(peak_filter)
    m.setPreProcessing(pre_processing)
    m.setLeftRightCheck(left_right_check)
    m.setLeftRightCheckThreshold(left_right_check_threshold)


cdef _compute(_Matcher* m, left, right):
    a, b = _grey_pair(left, right)
    cdef unsigned char[:, ::1] va = a
    cdef unsigned char[:, ::1] vb = b
    cdef int h = a.shape[0]
    cdef int w = a.shape[1]
    with nogil:
        m.compute(&va[0, 0], &vb[0, 0], w, h)


cdef object _disparity(_Matcher* m):
    import numpy as np
    if not m.computed():
        raise RuntimeError("compute() is required first")
    out = np.empty((m.height(), m.width()), dtype=np.float32)
    cdef float[:, ::1] view = out
    m.disparity(&view[0, 0])
    return out


cdef object _cloud(_Matcher* m, float u_c, float v_c, float focal,
                   float baseline, rgb):
    import numpy as np
    if not m.computed():
        raise RuntimeError("compute() is required first")
    if not focal > 0 or not baseline > 0:
        raise ValueError("focal and baseline must be > 0")
    cdef PointCloud xyz
    cdef PointCloud_PointXYZRGB coloured
    cdef cPointCloud[PointXYZ]* pxyz
    cdef cPointCloud[PointXYZRGB]* prgb
    cdef unsigned char[:, :, ::1] texture
    if rgb is None:
        xyz = PointCloud()
        pxyz = xyz.ptr()
        with nogil:
            m.pointCloud(u_c, v_c, focal, baseline, deref(pxyz))
        return xyz
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.shape != (m.height(), m.width(), 3):
        raise ValueError("rgb must be (%d, %d, 3) uint8"
                         % (m.height(), m.width()))
    texture = arr
    coloured = PointCloud_PointXYZRGB()
    prgb = coloured.ptr()
    with nogil:
        m.colourPointCloud(u_c, v_c, focal, baseline, &texture[0, 0, 0],
                           deref(prgb))
    return coloured


cdef class BlockBasedStereoMatching:
    """Sum-of-absolute-differences block matching
    (pcl::BlockBasedStereoMatching) over a ``(2 radius + 1)^2`` window.

    Disparities run from `x_offset` to `x_offset + max_disparity`; a
    pixel `ratio_filter` / `peak_filter` find ambiguous, or that fails
    the left-right check, is invalid. The first `max_disparity` columns
    (and the window's border) have no disparity.
    """

    cdef BlockStereo* me

    def __cinit__(self, int max_disparity=60, int radius=5, int x_offset=0,
                  int ratio_filter=0, int peak_filter=0,
                  bint pre_processing=False, bint left_right_check=False,
                  int left_right_check_threshold=1):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me = new BlockStereo()
        _configure(self.me, max_disparity, x_offset, ratio_filter,
                   peak_filter, pre_processing, left_right_check,
                   left_right_check_threshold)
        self.me.setRadius(radius)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_MaxDisparity(self, int d):
        if d < 1:
            raise ValueError("max_disparity must be >= 1")
        self.me.setMaxDisparity(d)

    def set_XOffset(self, int x):
        self.me.setXOffset(x)

    def set_RatioFilter(self, int r):
        self.me.setRatioFilter(r)

    def set_PeakFilter(self, int p):
        self.me.setPeakFilter(p)

    def set_PreProcessing(self, bint on):
        self.me.setPreProcessing(on)

    def set_LeftRightCheck(self, bint on):
        self.me.setLeftRightCheck(on)

    def set_LeftRightCheckThreshold(self, int t):
        self.me.setLeftRightCheckThreshold(t)

    def set_Radius(self, int radius):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me.setRadius(radius)

    def compute(self, left, right):
        """Match a rectified pair: `left` is the reference image."""
        _compute(self.me, left, right)

    def median_filter(self, int radius):
        """Median of each valid pixel's valid neighbours (PCL's
        medianFilter); the `radius`-wide border becomes invalid."""
        if not self.me.computed():
            raise RuntimeError("compute() is required first")
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me.medianFilter(radius)

    def get_DisparityMap(self):
        return _disparity(self.me)

    def get_PointCloud(self, float u_c, float v_c, float focal,
                       float baseline, rgb=None):
        """An organized cloud (NaN where invalid); with `rgb` (h, w, 3)
        uint8, a PointCloud_PointXYZRGB."""
        return _cloud(self.me, u_c, v_c, focal, baseline, rgb)


cdef class AdaptiveCostSOStereoMatching:
    """Adaptive-weight cost aggregation with scanline optimisation
    (pcl::AdaptiveCostSOStereoMatching): window weights fall off with
    distance (`gamma_s`) and colour difference (`gamma_c`), and a
    disparity jump costs `smooth_weak` (one level) or `smooth_strong`.
    Otherwise as :class:`BlockBasedStereoMatching`.
    """

    cdef AdaptiveStereo* me

    def __cinit__(self, int max_disparity=60, int radius=5, int gamma_s=15,
                  int gamma_c=25, int smooth_weak=20, int smooth_strong=100,
                  int x_offset=0, int ratio_filter=0, int peak_filter=0,
                  bint pre_processing=False, bint left_right_check=False,
                  int left_right_check_threshold=1):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me = new AdaptiveStereo()
        _configure(self.me, max_disparity, x_offset, ratio_filter,
                   peak_filter, pre_processing, left_right_check,
                   left_right_check_threshold)
        self.me.setRadius(radius)
        self.me.setGammaS(gamma_s)
        self.me.setGammaC(gamma_c)
        self.me.setSmoothWeak(smooth_weak)
        self.me.setSmoothStrong(smooth_strong)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_MaxDisparity(self, int d):
        if d < 1:
            raise ValueError("max_disparity must be >= 1")
        self.me.setMaxDisparity(d)

    def set_XOffset(self, int x):
        self.me.setXOffset(x)

    def set_RatioFilter(self, int r):
        self.me.setRatioFilter(r)

    def set_PeakFilter(self, int p):
        self.me.setPeakFilter(p)

    def set_PreProcessing(self, bint on):
        self.me.setPreProcessing(on)

    def set_LeftRightCheck(self, bint on):
        self.me.setLeftRightCheck(on)

    def set_LeftRightCheckThreshold(self, int t):
        self.me.setLeftRightCheckThreshold(t)

    def set_Radius(self, int radius):
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me.setRadius(radius)

    def set_GammaS(self, int g):
        self.me.setGammaS(g)

    def set_GammaC(self, int g):
        self.me.setGammaC(g)

    def set_SmoothWeak(self, int s):
        self.me.setSmoothWeak(s)

    def set_SmoothStrong(self, int s):
        self.me.setSmoothStrong(s)

    def compute(self, left, right):
        _compute(self.me, left, right)

    def median_filter(self, int radius):
        if not self.me.computed():
            raise RuntimeError("compute() is required first")
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.me.medianFilter(radius)

    def get_DisparityMap(self):
        return _disparity(self.me)

    def get_PointCloud(self, float u_c, float v_c, float focal,
                       float baseline, rgb=None):
        return _cloud(self.me, u_c, v_c, focal, baseline, rgb)


cdef _disparity_input(disparity, center, float focal, float baseline):
    import numpy as np
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        raise ValueError("disparity must be a non-empty (h, w) array")
    if not focal > 0 or not baseline > 0:
        raise ValueError("focal and baseline must be > 0")
    h, w = d.shape
    cx, cy = (w / 2.0, h / 2.0) if center is None else center
    return d, w, h, float(cx), float(cy)


cdef _rgb_input(image, size_t w, size_t h):
    import numpy as np
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    if arr.shape != (h, w, 3):
        raise ValueError("image must be (%d, %d, 3) uint8" % (h, w))
    return arr


def disparity_to_cloud(disparity, float focal, float baseline, center=None,
                       float min_disparity=0.0, max_disparity=None,
                       image=None):
    """An organized cloud from a disparity map in pixels
    (pcl::DisparityMapConverter): ``z = focal * baseline / d`` for each
    pixel with ``min_disparity < d < max_disparity``, NaN elsewhere.
    `center` is the principal point (default the image centre). With
    `image` (h, w, 3) uint8 the result is a PointCloud_PointXYZI whose
    intensity is the pixel's mean of r, g, b."""
    import numpy as np
    d, w, h, cx, cy = _disparity_input(disparity, center, focal, baseline)
    cdef float top = np.inf if max_disparity is None else max_disparity
    cdef vector[float] flat = d.ravel().tolist()
    cdef PointCloud xyz
    cdef PointCloud_PointXYZI xyzi
    cdef unsigned char[:, :, ::1] texture
    if image is None:
        xyz = PointCloud()
        disparityToCloud(flat, w, h, cx, cy, focal, baseline, min_disparity,
                         top, deref(xyz.ptr()))
        return xyz
    texture = _rgb_input(image, w, h)
    xyzi = PointCloud_PointXYZI()
    disparityToIntensityCloud(flat, w, h, cx, cy, focal, baseline,
                              min_disparity, top, &texture[0, 0, 0],
                              deref(xyzi.ptr()))
    return xyzi


def digital_elevation_map(disparity, image, float focal, float baseline,
                          center=None, min_disparity=None,
                          max_disparity=None, size_t columns=64,
                          size_t rows=32, size_t min_points_in_cell=1):
    """A (rows, columns, 6) float32 grid
    (pcl::DigitalElevationMapBuilder): cells are image-column bins by
    disparity bins (nearest last), each holding x, height y, z,
    intensity, intensity variance and height variance; NaN where fewer
    than `min_points_in_cell` pixels fell in. PCL's "mean" height and
    intensity are the centre of the cell histogram's (smoothed) mode bin.

    PCL bins heights over y = -0.5 .. 1.5 (camera frame, y down) at 1 cm
    and that range is fixed, so a scene in other units or at another
    camera height gets clamped heights. Each cell's x and z come from
    the lower edge of its disparity bin, and PCL leaves a point at the
    camera for a disparity of 0, so `min_disparity` defaults to just
    below the smallest finite disparity (`max_disparity` to just above
    the largest) rather than 0."""
    import numpy as np
    d, w, h, cx, cy = _disparity_input(disparity, center, focal, baseline)
    if columns < 1 or rows < 1:
        raise ValueError("columns and rows must be >= 1")
    cdef float top
    cdef float bottom
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size == 0 and (min_disparity is None or max_disparity is None):
        raise ValueError("the disparity map has no positive finite value")
    if max_disparity is None:
        top = float(np.nextafter(np.float32(finite.max()), np.float32(np.inf)))
    else:
        top = max_disparity
    if min_disparity is None:
        bottom = float(np.nextafter(np.float32(finite.min()), np.float32(0)))
    else:
        bottom = min_disparity
    if not top > bottom:
        raise ValueError("max_disparity must exceed min_disparity")
    cdef unsigned char[:, :, ::1] texture = _rgb_input(image, w, h)
    cdef vector[float] flat = d.ravel().tolist()
    cdef vector[float] cells
    elevationMap(flat, w, h, cx, cy, focal, baseline, bottom, top,
                 columns, rows, min_points_in_cell, &texture[0, 0, 0], cells)
    return np.asarray(cells, dtype=np.float32).reshape(rows, columns, 6)
