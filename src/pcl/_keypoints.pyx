# distutils: language = c++
# cython: language_level=3
"""Keypoint wrappers, named after sirokujira/python-pcl
(pcl/pxi/KeyPoint/).

Reach `HarrisKeypoint3D` through `PointCloud.make_HarrisKeypoint3D()`.
python-pcl's other entry in this group, `UniformSampling`, lives in
`pcl._filters` here: PCL moved it out of keypoints/ in 1.9 and it is a
filter now.

`compute()` returns an ``(n, 4)`` float32 array — x, y, z, response —
rather than a cloud object. PCL puts the Harris response in the output
point's intensity field, and the response is the reason to run a
detector at all, so handing back the pair together beats a cloud whose
fourth column you have to know to look in. Same call the
`NormalEstimation` wrapper makes.
"""

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointXYZ, PointXYZI
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.keypoints.harris_3d cimport HarrisKeypoint3D as cHarrisKeypoint3D
from pcl.pxd.compat.keypoint_args cimport (
    AGAST_5_8, AGAST_7_12S, HARRIS_METHOD_CURVATURE, HARRIS_METHOD_HARRIS,
    HARRIS_METHOD_LOWE, HARRIS_METHOD_NOBLE, HARRIS_METHOD_TOMASI, OAST_9_16,
    agastDetect, setHarrisMethod)

from pcl._pointcloud cimport PointCloud
from pcl.pxd.point_types cimport Normal, PointXYZRGB, PointWithScale
from pcl.pxd.keypoints.iss_3d cimport ISSKeypoint3D as cISSKeypoint3D
from pcl.pxd.keypoints.sift_keypoint cimport SIFTKeypoint as cSIFTKeypoint
from pcl.pxd.compat.feature_types cimport SUSANKeypointRGB as cSUSANKeypointRGB
from pcl._pointtypes cimport PointCloud_Normal, PointCloud_PointXYZRGB
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uint8_t
from pcl.pxd.point_types cimport BRISKSignature512, PointUV, PointXYZRGBA
from pcl.pxd.keypoints.harris_2d cimport HarrisKeypoint2D as cHarrisKeypoint2D
from pcl.pxd.keypoints.trajkovic_2d cimport (
    TrajkovicKeypoint2D as cTrajkovicKeypoint2D)
from pcl.pxd.keypoints.trajkovic_3d cimport (
    TrajkovicKeypoint3D as cTrajkovicKeypoint3D)
from pcl.pxd.keypoints.agast_2d cimport AgastKeypoint2D as cAgastKeypoint2D
from pcl.pxd.keypoints.brisk_2d cimport BriskKeypoint2D as cBriskKeypoint2D
from pcl.pxd.keypoints.smoothed_surfaces_keypoint cimport (
    SmoothedSurfacesKeypoint as cSmoothedSurfacesKeypoint)
from pcl.pxd.features.brisk_2d cimport BRISK2DEstimation as cBRISK2DEstimation
from pcl.pxd.compat.keypoint_args cimport (
    TRAJKOVIC_EIGHT_CORNERS, TRAJKOVIC_FOUR_CORNERS, addSmoothedCloud,
    setHarris2DMethod, setTrajkovic2DMethod, setTrajkovic3DMethod)
from pcl._pointtypes cimport PointCloud_PointXYZRGBA


# ResponseMethod is an enum nested in a class template, which Cython
# cannot name; the values come from pcl/compat/keypoint_args.h so they
# stay tied to the header rather than being copied here.
HARRIS = HARRIS_METHOD_HARRIS
NOBLE = HARRIS_METHOD_NOBLE
LOWE = HARRIS_METHOD_LOWE
TOMASI = HARRIS_METHOD_TOMASI
CURVATURE = HARRIS_METHOD_CURVATURE

# TrajkovicKeypoint2D / 3D ComputationMethod, through the same shim.
FOUR_CORNERS = TRAJKOVIC_FOUR_CORNERS
EIGHT_CORNERS = TRAJKOVIC_EIGHT_CORNERS


cdef class HarrisKeypoint3D:
    """Harris corner detector for point clouds (pcl::HarrisKeypoint3D).

    `set_Radius` is required — it is the neighbourhood the response is
    computed over, and PCL's default of 0.01 is smaller than most real
    clouds want.
    """

    cdef cHarrisKeypoint3D[PointXYZ, PointXYZI]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cHarrisKeypoint3D[PointXYZ, PointXYZI]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_Radius(self, float radius):
        self.me.setRadius(radius)

    def set_RadiusSearch(self, float radius):
        """python-pcl spelling of `set_Radius`."""
        self.me.setRadius(radius)

    def set_Threshold(self, float threshold):
        self.me.setThreshold(threshold)

    def set_NonMaxSupression(self, bint suppress):
        """Keep only local maxima. (PCL's spelling of `suppression`.)"""
        self.me.setNonMaxSupression(suppress)

    def set_Refine(self, bint refine):
        self.me.setRefine(refine)

    def set_Method(self, int method):
        """One of `pcl.HARRIS`, `NOBLE`, `LOWE`, `TOMASI`, `CURVATURE`."""
        setHarrisMethod(deref(self.me), method)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, 4)`` float32 array: x, y, z, response."""
        import numpy as np
        cdef cPointCloud[PointXYZI] out
        with nogil:
            self.me.compute(out)

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PointXYZI* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.intensity
        return result


cdef class ISSKeypoint3D:
    """Intrinsic Shape Signatures keypoints (pcl::ISSKeypoint3D).

    A point is a keypoint when the eigenvalues of its neighbourhood's
    scatter matrix are well separated (`set_Threshold21` /
    `set_Threshold32` are the largest allowed ratios) and it is the
    strongest within `set_NonMaxRadius`. `set_SalientRadius` sets the
    neighbourhood; both radii are in the cloud's own units. Normals are
    only needed with a border radius. Returns the keypoints as a
    :class:`PointCloud`.
    """

    cdef cISSKeypoint3D[PointXYZ, PointXYZ, Normal]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cISSKeypoint3D[PointXYZ, PointXYZ, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_SalientRadius(self, double radius):
        if radius <= 0:
            raise ValueError("salient radius must be positive")
        self.me.setSalientRadius(radius)

    def set_NonMaxRadius(self, double radius):
        if radius <= 0:
            raise ValueError("non-max radius must be positive")
        self.me.setNonMaxRadius(radius)

    def set_NormalRadius(self, double radius):
        self.me.setNormalRadius(radius)

    def set_BorderRadius(self, double radius):
        self.me.setBorderRadius(radius)

    def set_Threshold21(self, double gamma):
        self.me.setThreshold21(gamma)

    def set_Threshold32(self, double gamma):
        self.me.setThreshold32(gamma)

    def set_MinNeighbors(self, int count):
        self.me.setMinNeighbors(count)

    def set_Normals(self, PointCloud_Normal normals not None):
        self.me.setNormals(normals.thisptr_shared)

    def set_AngleThreshold(self, float angle):
        self.me.setAngleThreshold(angle)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return the keypoints as a :class:`PointCloud`."""
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.compute(deref(out))
        return pc


cdef class SIFTKeypoint:
    """SIFT keypoints over a 3-D scale space (pcl::SIFTKeypoint) on a
    :class:`PointCloud_PointXYZRGB`, whose luminance is the "intensity".

    `set_Scales(min_scale, n_octaves, n_scales_per_octave)` is required
    (the smallest scale is in the cloud's own units); `set_MinimumContrast`
    discards weak responses. `compute()` returns an ``(n, 7)`` array:
    x, y, z, scale, angle, response, octave.
    """

    cdef cSIFTKeypoint[PointXYZRGB, PointWithScale]* me
    cdef bint has_scales

    def __cinit__(self, PointCloud_PointXYZRGB pc=None):
        self.me = new cSIFTKeypoint[PointXYZRGB, PointWithScale]()
        self.has_scales = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_Scales(self, float min_scale, int n_octaves,
                   int n_scales_per_octave):
        if min_scale <= 0 or n_octaves < 1 or n_scales_per_octave < 1:
            raise ValueError(
                "min_scale must be positive and both counts at least 1")
        self.me.setScales(min_scale, n_octaves, n_scales_per_octave)
        self.has_scales = True

    def set_MinimumContrast(self, float contrast):
        self.me.setMinimumContrast(contrast)

    def compute(self):
        """Return an ``(n, 7)`` float32 array: x, y, z, scale, angle,
        response, octave."""
        import numpy as np
        # Without scales PCL loops zero octaves and returns nothing.
        if not self.has_scales:
            raise RuntimeError(
                "set_Scales(min_scale, n_octaves, n_scales_per_octave) is "
                "required before compute()")
        cdef cPointCloud[PointWithScale] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 7), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PointWithScale* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.scale
            view[i, 4] = p.angle
            view[i, 5] = p.response
            view[i, 6] = <float> p.octave
        return result


cdef class SUSANKeypoint:
    """SUSAN corner keypoints (pcl::SUSANKeypoint) on a
    :class:`PointCloud_PointXYZRGB`.

    A point is a corner when fewer than half of its neighbours within
    `set_Radius` are "similar" to it, where similar means EITHER an
    intensity within `set_IntensityThreshold` OR a normal within
    `set_AngularThreshold` -- the two tests are OR-ed. So on flat
    geometry every neighbour passes the normal test and no colour corner
    is ever found: set the angular threshold negative to detect by colour
    alone (or the intensity threshold negative for geometry alone).
    Normals are estimated internally unless `set_Normals` is called, and
    non-maximum suppression can discard everything when responses tie,
    as they do on synthetic data. Returns the keypoints as a
    :class:`PointCloud`.
    """

    cdef cSUSANKeypointRGB* me

    def __cinit__(self, PointCloud_PointXYZRGB pc=None):
        self.me = new cSUSANKeypointRGB()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_Radius(self, float radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadius(radius)

    def set_DistanceThreshold(self, float threshold):
        self.me.setDistanceThreshold(threshold)

    def set_AngularThreshold(self, float threshold):
        self.me.setAngularThreshold(threshold)

    def set_IntensityThreshold(self, float threshold):
        self.me.setIntensityThreshold(threshold)

    def set_Normals(self, PointCloud_Normal normals not None):
        self.me.setNormals(normals.thisptr_shared)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def set_NonMaxSupression(self, bint suppress):
        self.me.setNonMaxSupression(suppress)

    def set_GeometricValidation(self, bint validate):
        self.me.setGeometricValidation(validate)

    def compute(self):
        """Return the keypoints as a :class:`PointCloud`."""
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.compute(deref(out))
        return pc


# --- helpers for the second keypoint batch -------------------------------

cdef object _xyzi_rows(cPointCloud[PointXYZI]& out):
    """Keypoints with their response as an ``(n, 4)`` float32 array."""
    import numpy as np
    cdef Py_ssize_t n = <Py_ssize_t> out.size()
    result = np.empty((n, 4), dtype=np.float32)
    cdef float[:, ::1] view = result
    cdef Py_ssize_t i
    for i in range(n):
        view[i, 0] = out[i].x
        view[i, 1] = out[i].y
        view[i, 2] = out[i].z
        view[i, 3] = out[i].intensity
    return result


cdef object _scale_rows(cPointCloud[PointWithScale]& out):
    """PointWithScale keypoints as SIFTKeypoint returns them."""
    import numpy as np
    cdef Py_ssize_t n = <Py_ssize_t> out.size()
    result = np.empty((n, 7), dtype=np.float32)
    cdef float[:, ::1] view = result
    cdef Py_ssize_t i
    for i in range(n):
        view[i, 0] = out[i].x
        view[i, 1] = out[i].y
        view[i, 2] = out[i].z
        view[i, 3] = out[i].scale
        view[i, 4] = out[i].angle
        view[i, 5] = out[i].response
        view[i, 6] = out[i].octave
    return result


cdef object _organized_rgba(PointCloud_PointXYZRGBA pc, str name):
    if pc.ptr().height < 2:
        raise ValueError(
            "%s works on the image of an organized cloud (height > 1); build "
            "one with PointCloud_PointXYZRGBA.from_organized_rgba_array"
            % name)


cdef class HarrisKeypoint2D:
    """Harris corners on the intensity image of an organized
    :class:`PointCloud_PointXYZRGBA` (pcl::HarrisKeypoint2D).
    `set_Method` takes ``pcl.HARRIS``, ``NOBLE``, ``LOWE`` or ``TOMASI``;
    windows and distances are in pixels. `compute()` returns ``(n, 4)``:
    x, y, z of the corner pixel, response.
    """

    cdef cHarrisKeypoint2D[PointXYZRGBA, PointXYZI]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cHarrisKeypoint2D[PointXYZRGBA, PointXYZI]()
        # It runs Keypoint::initCompute, which refuses to start without a
        # neighbour-search parameter the 2-D detector never uses.
        self.me.setKSearch(1)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_rgba(pc, "HarrisKeypoint2D")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_Method(self, int method):
        if method not in (HARRIS, NOBLE, LOWE, TOMASI):
            raise ValueError("method must be pcl.HARRIS, NOBLE, LOWE or "
                             "TOMASI for the 2-D detector")
        setHarris2DMethod(deref(self.me), method)

    def set_WindowWidth(self, int pixels):
        self.me.setWindowWidth(pixels)

    def set_WindowHeight(self, int pixels):
        self.me.setWindowHeight(pixels)

    def set_SkippedPixels(self, int pixels):
        self.me.setSkippedPixels(pixels)

    def set_MinimalDistance(self, int pixels):
        self.me.setMinimalDistance(pixels)

    def set_Threshold(self, float threshold):
        self.me.setThreshold(threshold)

    def set_NonMaxSupression(self, bint suppress):
        self.me.setNonMaxSupression(suppress)

    def set_Refine(self, bint refine):
        self.me.setRefine(refine)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef cPointCloud[PointXYZI] out
        with nogil:
            self.me.compute(out)
        return _xyzi_rows(out)


cdef class TrajkovicKeypoint2D:
    """Trajkovic and Hedley's fast corners on the intensity image of an
    organized :class:`PointCloud_PointXYZRGBA`
    (pcl::TrajkovicKeypoint2D). `set_Method` takes ``pcl.FOUR_CORNERS``
    (default) or ``EIGHT_CORNERS``. `compute()` returns ``(n, 4)``: x, y,
    z, cornerness.
    """

    cdef cTrajkovicKeypoint2D[PointXYZRGBA, PointXYZI]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cTrajkovicKeypoint2D[PointXYZRGBA, PointXYZI]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_rgba(pc, "TrajkovicKeypoint2D")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_Method(self, int method):
        if method not in (FOUR_CORNERS, EIGHT_CORNERS):
            raise ValueError("method must be pcl.FOUR_CORNERS or EIGHT_CORNERS")
        setTrajkovic2DMethod(deref(self.me), method)

    def set_WindowSize(self, int pixels):
        if pixels < 3 or pixels % 2 == 0:
            raise ValueError("window size must be odd and at least 3")
        self.me.setWindowSize(pixels)

    def get_WindowSize(self):
        return self.me.getWindowSize()

    def set_FirstThreshold(self, float threshold):
        self.me.setFirstThreshold(threshold)

    def get_FirstThreshold(self):
        return self.me.getFirstThreshold()

    def set_SecondThreshold(self, float threshold):
        self.me.setSecondThreshold(threshold)

    def get_SecondThreshold(self):
        return self.me.getSecondThreshold()

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef cPointCloud[PointXYZI] out
        with nogil:
            self.me.compute(out)
        return _xyzi_rows(out)


cdef class TrajkovicKeypoint3D:
    """Trajkovic corners on the NORMALS of an organized cloud
    (pcl::TrajkovicKeypoint3D): where the normal field changes in every
    direction. Needs the cloud's normals on the same grid. `compute()`
    returns ``(n, 4)``: x, y, z, cornerness.
    """

    cdef cTrajkovicKeypoint3D[PointXYZ, PointXYZI]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cTrajkovicKeypoint3D[PointXYZ, PointXYZI]()
        self.n_input = -1
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if pc.ptr().height < 2:
            raise ValueError("TrajkovicKeypoint3D needs an organized cloud "
                             "(height > 1)")
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_Normals(self, PointCloud_Normal normals not None):
        if self.n_input >= 0 and normals.size != self.n_input:
            raise ValueError("normals must match the cloud point for point")
        self.me.setNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_Method(self, int method):
        if method not in (FOUR_CORNERS, EIGHT_CORNERS):
            raise ValueError("method must be pcl.FOUR_CORNERS or EIGHT_CORNERS")
        setTrajkovic3DMethod(deref(self.me), method)

    def set_WindowSize(self, int pixels):
        if pixels < 3 or pixels % 2 == 0:
            raise ValueError("window size must be odd and at least 3")
        self.me.setWindowSize(pixels)

    def get_WindowSize(self):
        return self.me.getWindowSize()

    def set_FirstThreshold(self, float threshold):
        self.me.setFirstThreshold(threshold)

    def get_FirstThreshold(self):
        return self.me.getFirstThreshold()

    def set_SecondThreshold(self, float threshold):
        self.me.setSecondThreshold(threshold)

    def get_SecondThreshold(self):
        return self.me.getSecondThreshold()

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud(...) is required first")
        if not self.has_normals:
            raise RuntimeError("set_Normals(...) is required first")
        cdef cPointCloud[PointXYZI] out
        with nogil:
            self.me.compute(out)
        return _xyzi_rows(out)


_AGAST_DETECTORS = {"7_12s": AGAST_7_12S, "5_8": AGAST_5_8,
                    "oast_9_16": OAST_9_16}


cdef class AgastKeypoint2D:
    """AGAST corners (pcl::AgastKeypoint2D, Mair et al.) on the intensity
    image of an organized :class:`PointCloud_PointXYZRGBA`. `compute()`
    returns ``(n, 2)`` float32 pixel coordinates (u = column, v = row).

    `detector` picks the mask: "7_12s" (PCL's default), "5_8" or
    "oast_9_16". PCL builds its detector once, from the first image's
    size and threshold, and keeps it -- a later frame of another size
    was read with the first one's row length (0 corners where there were
    4) and a changed threshold ignored -- so each `compute()` here gets a
    fresh one (compat/keypoint_args.h).
    """

    cdef cAgastKeypoint2D[PointXYZRGBA, PointUV]* me
    cdef bint has_input
    cdef int kind

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None, str detector="7_12s"):
        if detector not in _AGAST_DETECTORS:
            raise ValueError("detector must be one of %s"
                             % ", ".join(_AGAST_DETECTORS))
        self.me = new cAgastKeypoint2D[PointXYZRGBA, PointUV]()
        self.has_input = False
        self.kind = _AGAST_DETECTORS[detector]
        if pc is not None:
            self.set_InputCloud(pc)

    def set_Detector(self, str detector):
        if detector not in _AGAST_DETECTORS:
            raise ValueError("detector must be one of %s"
                             % ", ".join(_AGAST_DETECTORS))
        self.kind = _AGAST_DETECTORS[detector]

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_rgba(pc, "AgastKeypoint2D")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_Threshold(self, double threshold):
        """Intensity difference a corner needs (PCL default 10)."""
        self.me.setThreshold(threshold)

    def get_Threshold(self):
        return self.me.getThreshold()

    def set_MaxKeypoints(self, unsigned int count):
        self.me.setMaxKeypoints(count)

    def get_MaxKeypoints(self):
        return self.me.getMaxKeypoints()

    def set_MaxDataValue(self, double value):
        """Largest intensity value (255 for 8-bit colour)."""
        self.me.setMaxDataValue(value)

    def get_MaxDataValue(self):
        return self.me.getMaxDataValue()

    def set_NonMaxSuppression(self, bint enabled):
        self.me.setNonMaxSuppression(enabled)

    def get_NonMaxSuppression(self):
        return self.me.getNonMaxSuppression()

    def compute(self):
        import numpy as np
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef cPointCloud[PointUV] out
        with nogil:
            agastDetect(deref(self.me), self.kind, out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 2), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i, 0] = out[i].u
            view[i, 1] = out[i].v
        return result


cdef class BriskKeypoint2D:
    """BRISK keypoints (pcl::BriskKeypoint2D, Leutenegger et al.): AGAST
    corners over a scale-space pyramid of the intensity image of an
    organized :class:`PointCloud_PointXYZRGBA`. `compute()` returns
    ``(n, 7)`` like `SIFTKeypoint`: x, y, z, scale, angle, response,
    octave -- where x and y are the refined PIXEL position (column, row),
    not 3-D coordinates. Pass that array straight to
    `BRISK2DEstimation.compute`.
    """

    cdef cBriskKeypoint2D[PointXYZRGBA, PointWithScale]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None, int octaves=4,
                  int threshold=60):
        self.me = new cBriskKeypoint2D[PointXYZRGBA, PointWithScale](
            octaves, threshold)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_rgba(pc, "BriskKeypoint2D")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_Threshold(self, int threshold):
        self.me.setThreshold(threshold)

    def get_Threshold(self):
        return self.me.getThreshold()

    def set_Octaves(self, int octaves):
        if octaves < 0:
            raise ValueError("octaves must be >= 0")
        self.me.setOctaves(octaves)

    def get_Octaves(self):
        return self.me.getOctaves()

    def set_RemoveInvalid3DKeypoints(self, bint remove):
        self.me.setRemoveInvalid3DKeypoints(remove)

    def get_RemoveInvalid3DKeypoints(self):
        return self.me.getRemoveInvalid3DKeypoints()

    def compute(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef cPointCloud[PointWithScale] out
        with nogil:
            self.me.compute(out)
        return _scale_rows(out)


cdef class BRISK2DEstimation:
    """BRISK binary descriptors (pcl::BRISK2DEstimation) at
    `BriskKeypoint2D` keypoints of the same organized image.

        keypoints = pcl.BriskKeypoint2D(image).compute()
        descriptors, kept = pcl.BRISK2DEstimation(image).compute(keypoints)

    `compute` returns ``(descriptors, keypoints)``: ``(k, 64)`` uint8
    descriptors (512 bits each) and the ``(k, 7)`` keypoints they belong
    to. PCL drops keypoints too close to the image border, so ``k`` can
    be smaller than the number passed in; the rows correspond.
    """

    cdef cBRISK2DEstimation[PointXYZRGBA, BRISKSignature512, PointWithScale]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cBRISK2DEstimation[PointXYZRGBA, BRISKSignature512,
                                         PointWithScale]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_rgba(pc, "BRISK2DEstimation")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_RotationInvariance(self, bint enable):
        self.me.setRotationInvariance(enable)

    def set_ScaleInvariance(self, bint enable):
        self.me.setScaleInvariance(enable)

    def compute(self, keypoints):
        import numpy as np
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        arr = np.ascontiguousarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError("keypoints must be the (n, 7) array "
                             "BriskKeypoint2D.compute() returns")
        cdef shared_ptr[cPointCloud[PointWithScale]] kp
        kp.reset(new cPointCloud[PointWithScale]())
        cdef float[:, ::1] view = arr
        cdef PointWithScale p
        cdef Py_ssize_t i, j
        for i in range(arr.shape[0]):
            p.x = view[i, 0]
            p.y = view[i, 1]
            p.z = view[i, 2]
            p.scale = view[i, 3]
            p.angle = view[i, 4]
            p.response = view[i, 5]
            p.octave = <int> view[i, 6]
            kp.get().push_back(p)
        self.me.setKeypoints(kp)
        cdef cPointCloud[BRISKSignature512] out
        with nogil:
            self.me.compute(out)
        descriptors = np.empty((out.size(), 64), dtype=np.uint8)
        cdef uint8_t[:, ::1] dview = descriptors
        for i in range(<Py_ssize_t> out.size()):
            for j in range(64):
                dview[i, j] = out[i].descriptor[j]
        return descriptors, _scale_rows(deref(kp.get()))


cdef class SmoothedSurfacesKeypoint:
    """Keypoints that persist across smoothed versions of a surface
    (pcl::SmoothedSurfacesKeypoint, Rusu et al.'s scale space).

        detector = pcl.SmoothedSurfacesKeypoint(cloud, normals, scale=0.01)
        for smoothed, smoothed_normals, scale in pyramid:
            detector.add_SmoothedPointCloud(smoothed, smoothed_normals, scale)
        keypoints = detector.compute()           # PointCloud

    Every cloud, the input included, must be the same points (the
    smoothing moves them, it does not resample), each with its normals.
    """

    cdef cSmoothedSurfacesKeypoint[PointXYZ, Normal]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals
    cdef int n_smoothed

    def __cinit__(self, PointCloud pc=None, PointCloud_Normal normals=None,
                  float scale=0.0):
        self.me = new cSmoothedSurfacesKeypoint[PointXYZ, Normal]()
        self.n_input = -1
        self.has_normals = False
        self.n_smoothed = 0
        if pc is not None:
            self.set_InputCloud(pc)
        if normals is not None:
            self.set_InputNormals(normals)
        if scale > 0:
            self.set_InputScale(scale)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_InputNormals(self, PointCloud_Normal normals not None):
        if self.n_input >= 0 and normals.size != self.n_input:
            raise ValueError("normals must match the input cloud")
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_InputScale(self, float scale):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.me.setInputScale(scale)

    def set_NeighborhoodConstant(self, float constant):
        self.me.setNeighborhoodConstant(constant)

    def get_NeighborhoodConstant(self):
        return self.me.getNeighborhoodConstant()

    def add_SmoothedPointCloud(self, PointCloud pc not None,
                               PointCloud_Normal normals not None,
                               float scale):
        if self.n_input >= 0 and pc.size != self.n_input:
            raise ValueError("every smoothed cloud must have the input's "
                             "points: %d vs %d" % (pc.size, self.n_input))
        if normals.size != pc.size:
            raise ValueError("normals must match the smoothed cloud")
        if scale <= 0:
            raise ValueError("scale must be positive")
        addSmoothedCloud(deref(self.me), pc.thisptr_shared,
                         normals.thisptr_shared, scale)
        self.n_smoothed += 1

    def reset_Clouds(self):
        self.me.resetClouds()
        self.n_smoothed = 0

    def compute(self):
        """Return the keypoints as a :class:`PointCloud`."""
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud(...) is required first")
        if not self.has_normals:
            raise RuntimeError("set_InputNormals(...) is required first")
        if self.n_smoothed == 0:
            raise RuntimeError("add_SmoothedPointCloud(...) at least once "
                               "first")
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.compute(deref(out))
        return pc
