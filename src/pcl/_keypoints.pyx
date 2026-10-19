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
    HARRIS_METHOD_CURVATURE, HARRIS_METHOD_HARRIS, HARRIS_METHOD_LOWE,
    HARRIS_METHOD_NOBLE, HARRIS_METHOD_TOMASI, setHarrisMethod)

from pcl._pointcloud cimport PointCloud
from pcl.pxd.point_types cimport Normal, PointXYZRGB, PointWithScale
from pcl.pxd.keypoints.iss_3d cimport ISSKeypoint3D as cISSKeypoint3D
from pcl.pxd.keypoints.sift_keypoint cimport SIFTKeypoint as cSIFTKeypoint
from pcl.pxd.compat.feature_types cimport SUSANKeypointRGB as cSUSANKeypointRGB
from pcl._pointtypes cimport PointCloud_Normal, PointCloud_PointXYZRGB


# ResponseMethod is an enum nested in a class template, which Cython
# cannot name; the values come from pcl/compat/keypoint_args.h so they
# stay tied to the header rather than being copied here.
HARRIS = HARRIS_METHOD_HARRIS
NOBLE = HARRIS_METHOD_NOBLE
LOWE = HARRIS_METHOD_LOWE
TOMASI = HARRIS_METHOD_TOMASI
CURVATURE = HARRIS_METHOD_CURVATURE


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
