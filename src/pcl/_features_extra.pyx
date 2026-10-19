# distutils: language = c++
# cython: language_level=3
"""The rest of pcl/features: OpenMP and colour variants of the
descriptors in `_features`, shape contexts, point-pair features, local
reference frames, intensity descriptors, organized edge detection, and
the mesh / multiscale helpers.

Each estimator follows the same shape as the ones in `_features`:

    estimator = pcl.FPFHEstimationOMP(cloud, threads=4)
    estimator.set_InputNormals(normals)
    estimator.set_RadiusSearch(0.05)
    descriptors = estimator.compute()      # (n, 33) float32

`compute()` always returns a float32 array with one row per input point
(per ordered PAIR of points for the point-pair features), so the result
can go straight into numpy, scikit-learn or `pcl.match_descriptors`.
"""

from cython.operator cimport dereference as deref

from libc.stdint cimport uint32_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport (
    CPPFSignature, FPFHSignature33, GFPFHSignature16, IntensityGradient,
    Label, Normal, PFHRGBSignature250, PPFRGBSignature,
    PPFSignature, PointXYZ, PointXYZI, PointXYZL, PointXYZRGBA, ReferenceFrame,
    SHOT1344, SHOT352, ShapeContext1980, UniqueShapeContext1960)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.point_indices cimport PointIndices
from pcl.pxd.vertices cimport Vertices
from pcl.pxd.features.fpfh_omp cimport FPFHEstimationOMP as cFPFHEstimationOMP
from pcl.pxd.features.shot_omp cimport (
    SHOTColorEstimationOMP as cSHOTColorEstimationOMP,
    SHOTEstimationOMP as cSHOTEstimationOMP)
from pcl.pxd.features.shot cimport SHOTColorEstimation as cSHOTColorEstimation
from pcl.pxd.features.pfhrgb cimport PFHRGBEstimation as cPFHRGBEstimation
from pcl.pxd.features.shape_context_3d cimport (
    ShapeContext3DEstimation as cShapeContext3DEstimation)
from pcl.pxd.features.usc cimport UniqueShapeContext as cUniqueShapeContext
from pcl.pxd.features.ppf cimport PPFEstimation as cPPFEstimation
from pcl.pxd.features.ppfrgb cimport PPFRGBEstimation as cPPFRGBEstimation
from pcl.pxd.features.cppf cimport CPPFEstimation as cCPPFEstimation
from pcl.pxd.features.shot_lrf cimport (
    SHOTLocalReferenceFrameEstimation as cSHOTLRF)
from pcl.pxd.features.shot_lrf_omp cimport (
    SHOTLocalReferenceFrameEstimationOMP as cSHOTLRFOMP)
from pcl.pxd.features.board cimport (
    BOARDLocalReferenceFrameEstimation as cBOARDLRF)
from pcl.pxd.features.flare cimport (
    FLARELocalReferenceFrameEstimation as cFLARELRF)
from pcl.pxd.features.intensity_gradient cimport (
    IntensityGradientEstimation as cIntensityGradientEstimation)
from pcl.pxd.features.intensity_spin cimport (
    IntensitySpinEstimation as cIntensitySpinEstimation)
from pcl.pxd.features.rift cimport RIFTEstimation as cRIFTEstimation
from pcl.pxd.features.linear_least_squares_normal cimport (
    LinearLeastSquaresNormalEstimation as cLinearLeastSquaresNormalEstimation)
from pcl.pxd.features.organized_edge_detection cimport (
    OrganizedEdgeBase as cOrganizedEdgeBase,
    OrganizedEdgeFromNormals as cOrganizedEdgeFromNormals,
    OrganizedEdgeFromRGB as cOrganizedEdgeFromRGB,
    OrganizedEdgeFromRGBNormals as cOrganizedEdgeFromRGBNormals)
from pcl.pxd.features.gfpfh cimport GFPFHEstimation as cGFPFHEstimation
from pcl.pxd.features.rops_estimation cimport ROPSEstimation as cROPSEstimation
from pcl.pxd.compat.feature_types cimport (
    Histogram135, Histogram20, Histogram32, approximateNormals,
    multiscaleInterestRegions, pairFeatures, persistentFPFH)
cimport pcl.pxd.compat.feature_types as ft

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointXYZI, PointCloud_PointXYZRGBA,
    wrap_normal_cloud)


# OrganizedEdgeBase::EDGELABEL_* bits (also the positions of the index
# lists the edge detectors return: bit 1 << k is list k).
EDGELABEL_NAN_BOUNDARY = ft.EDGELABEL_NAN_BOUNDARY
EDGELABEL_OCCLUDING = ft.EDGELABEL_OCCLUDING
EDGELABEL_OCCLUDED = ft.EDGELABEL_OCCLUDED
EDGELABEL_HIGH_CURVATURE = ft.EDGELABEL_HIGH_CURVATURE
EDGELABEL_RGB_CANNY = ft.EDGELABEL_RGB_CANNY

# pcl::NormType, for MultiscaleFeaturePersistence.set_DistanceMetric.
NORM_L1 = ft.NORM_L1
NORM_L2_SQR = ft.NORM_L2_SQR
NORM_L2 = ft.NORM_L2
NORM_LINF = ft.NORM_LINF
NORM_JM = ft.NORM_JM
NORM_B = ft.NORM_B
NORM_SUBLINEAR = ft.NORM_SUBLINEAR
NORM_CS = ft.NORM_CS
NORM_DIV = ft.NORM_DIV
NORM_PF = ft.NORM_PF
NORM_K = ft.NORM_K
NORM_KL = ft.NORM_KL
NORM_HIK = ft.NORM_HIK


# --- helpers -------------------------------------------------------------

cdef object _rows(Py_ssize_t n, Py_ssize_t width):
    import numpy as np
    return np.empty((n, width), dtype=np.float32)


cdef inline void _copy(float[:, ::1] view, Py_ssize_t i, const float* src,
                       Py_ssize_t width) noexcept nogil:
    cdef Py_ssize_t j
    for j in range(width):
        view[i, j] = src[j]


cdef object _need(bint ok, str what):
    if not ok:
        raise RuntimeError(what)


cdef object _check_normals(Py_ssize_t n_cloud, PointCloud_Normal normals):
    if normals.size != n_cloud:
        raise ValueError(
            "normals must match the cloud: %d points, %d normals"
            % (n_cloud, normals.size))


cdef object _frames(cPointCloud[ReferenceFrame]& out):
    """Reference frames as an ``(n, 9)`` array: x, y, z axis, row-major."""
    result = _rows(<Py_ssize_t> out.size(), 9)
    cdef float[:, ::1] view = result
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t> out.size()):
        _copy(view, i, &out[i].rf[0], 9)
    return result


# --- OpenMP variants -----------------------------------------------------

cdef class FPFHEstimationOMP:
    """FPFH over several threads (pcl::FPFHEstimationOMP): the same
    ``(n, 33)`` histograms as `FPFHEstimation`. `threads` 0 means all
    cores."""

    cdef cFPFHEstimationOMP[PointXYZ, Normal, FPFHSignature33]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None, unsigned int threads=0):
        self.me = new cFPFHEstimationOMP[PointXYZ, Normal, FPFHSignature33](
            threads)
        self.n_input = -1
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, 33)`` float32 array."""
        _need(self.n_input >= 0, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[FPFHSignature33] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 33)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].histogram[0], 33)
        return result


cdef class SHOTEstimationOMP:
    """SHOT over several threads (pcl::SHOTEstimationOMP): the same
    ``(n, 352)`` descriptors as `SHOTEstimation`."""

    cdef cSHOTEstimationOMP[PointXYZ, Normal, SHOT352]* me
    cdef bint has_input
    cdef bint has_normals
    cdef bint has_radius

    def __cinit__(self, PointCloud pc=None, unsigned int threads=0):
        self.me = new cSHOTEstimationOMP[PointXYZ, Normal, SHOT352](threads)
        self.has_input = False
        self.has_normals = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_LRFRadius(self, float radius):
        self.me.setLRFRadius(radius)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, 352)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first: "
              "SHOT has no k-neighbour form")
        cdef cPointCloud[SHOT352] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 352)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].descriptor[0], 352)
        return result


cdef class SHOTColorEstimation:
    """SHOT over shape AND colour (pcl::SHOTColorEstimation): 352 shape
    bins plus 992 CIELab colour bins per point, on a
    :class:`PointCloud_PointXYZRGBA`. Either half can be switched off;
    PCL then packs the other at the front of its 1344-value output, so
    `compute()` returns just the meaningful columns: 1344, 352 (shape
    only, identical to `SHOTEstimation`) or 992 (colour only). Each row
    is unit length as a whole, so with both halves the shape part is the
    SHOT descriptor scaled down, not equal to it."""

    cdef cSHOTColorEstimation[PointXYZRGBA, Normal, SHOT1344]* me
    cdef Py_ssize_t width
    cdef bint has_input
    cdef bint has_normals
    cdef bint has_radius

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None,
                  bint describe_shape=True, bint describe_color=True):
        if not (describe_shape or describe_color):
            raise ValueError("describe at least one of shape and colour")
        self.me = new cSHOTColorEstimation[PointXYZRGBA, Normal, SHOT1344](
            describe_shape, describe_color)
        self.width = 352 * describe_shape + 992 * describe_color
        self.has_input = False
        self.has_normals = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_LRFRadius(self, float radius):
        self.me.setLRFRadius(radius)

    def compute(self):
        """Return an ``(n, width)`` float32 array: 1344, 352 or 992
        columns depending on which halves are described."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[SHOT1344] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), self.width)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].descriptor[0], self.width)
        return result


cdef class SHOTColorEstimationOMP:
    """`SHOTColorEstimation` over several threads
    (pcl::SHOTColorEstimationOMP); same output widths."""

    cdef cSHOTColorEstimationOMP[PointXYZRGBA, Normal, SHOT1344]* me
    cdef Py_ssize_t width
    cdef bint has_input
    cdef bint has_normals
    cdef bint has_radius

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None,
                  bint describe_shape=True, bint describe_color=True,
                  unsigned int threads=0):
        if not (describe_shape or describe_color):
            raise ValueError("describe at least one of shape and colour")
        self.me = new cSHOTColorEstimationOMP[PointXYZRGBA, Normal, SHOT1344](
            describe_shape, describe_color, threads)
        self.width = 352 * describe_shape + 992 * describe_color
        self.has_input = False
        self.has_normals = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_LRFRadius(self, float radius):
        self.me.setLRFRadius(radius)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, width)`` float32 array: 1344, 352 or 992
        columns depending on which halves are described."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[SHOT1344] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), self.width)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].descriptor[0], self.width)
        return result


cdef class PFHRGBEstimation:
    """PFH plus three colour-ratio features (pcl::PFHRGBEstimation):
    ``(n, 250)`` on a :class:`PointCloud_PointXYZRGBA`."""

    cdef cPFHRGBEstimation[PointXYZRGBA, Normal, PFHRGBSignature250]* me
    cdef bint has_input
    cdef bint has_normals

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cPFHRGBEstimation[PointXYZRGBA, Normal,
                                        PFHRGBSignature250]()
        self.has_input = False
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def compute(self):
        """Return an ``(n, 250)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[PFHRGBSignature250] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 250)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].histogram[0], 250)
        return result


# --- shape contexts ------------------------------------------------------

cdef class ShapeContext3DEstimation:
    """3-D Shape Context (pcl::ShapeContext3DEstimation, Frome et al.): a
    spherical grid around each point's normal -- radial shells from
    `minimal_radius` out to the search radius, 12 azimuth x 11 elevation
    sectors -- each counting the neighbours weighted by local density.
    ``(n, 1980)``. The azimuth origin is arbitrary (`random=True` picks it
    at random, as the paper does), so matching has to try rotations;
    `UniqueShapeContext` fixes that with a local reference frame.
    """

    cdef cShapeContext3DEstimation[PointXYZ, Normal, ShapeContext1980]* me
    cdef bint has_input
    cdef bint has_normals
    cdef double radius

    def __cinit__(self, PointCloud pc=None, bint random=False):
        self.me = new cShapeContext3DEstimation[PointXYZ, Normal,
                                                ShapeContext1980](random)
        self.has_input = False
        self.has_normals = False
        self.radius = 0.0
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        """Outer radius of the grid."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.radius = radius

    def set_MinimalRadius(self, double radius):
        """Inner radius of the first shell (PCL default 0.1)."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setMinimalRadius(radius)

    def get_MinimalRadius(self):
        return self.me.getMinimalRadius()

    def set_PointDensityRadius(self, double radius):
        """Radius used to count a neighbour's local density (PCL default
        0.2)."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setPointDensityRadius(radius)

    def get_PointDensityRadius(self):
        return self.me.getPointDensityRadius()

    def compute(self):
        """Return an ``(n, 1980)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        if self.radius <= self.me.getMinimalRadius():
            raise ValueError(
                "the search radius (%g) must exceed the minimal radius (%g); "
                "PCL would only log that and return nothing"
                % (self.radius, self.me.getMinimalRadius()))
        cdef cPointCloud[ShapeContext1980] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 1980)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].descriptor[0], 1980)
        return result


cdef class UniqueShapeContext:
    """Unique Shape Context (pcl::UniqueShapeContext, Tombari et al.):
    3DSC computed in a repeatable local reference frame (radius
    `local_radius`), so a single descriptor per point is enough and no
    normals are needed. ``(n, 1960)``."""

    cdef cUniqueShapeContext[PointXYZ, UniqueShapeContext1960]* me
    cdef bint has_input
    cdef double radius

    def __cinit__(self, PointCloud pc=None):
        self.me = new cUniqueShapeContext[PointXYZ, UniqueShapeContext1960]()
        self.has_input = False
        self.radius = 0.0
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.radius = radius

    def set_MinimalRadius(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setMinimalRadius(radius)

    def get_MinimalRadius(self):
        return self.me.getMinimalRadius()

    def set_PointDensityRadius(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setPointDensityRadius(radius)

    def get_PointDensityRadius(self):
        return self.me.getPointDensityRadius()

    def set_LocalRadius(self, double radius):
        """Radius of the neighbourhood the reference frame comes from."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setLocalRadius(radius)

    def get_LocalRadius(self):
        return self.me.getLocalRadius()

    def compute(self):
        """Return an ``(n, 1960)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        if self.radius <= self.me.getMinimalRadius():
            raise ValueError(
                "the search radius (%g) must exceed the minimal radius (%g)"
                % (self.radius, self.me.getMinimalRadius()))
        cdef cPointCloud[UniqueShapeContext1960] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 1960)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].descriptor[0], 1960)
        return result


# --- point pair features -------------------------------------------------

cdef class PPFEstimation:
    """Point Pair Features (pcl::PPFEstimation, Drost et al.) of EVERY
    ordered pair of input points: ``(n * n, 5)`` rows of
    ``f1, f2, f3, f4, alpha_m``, pair ``(i, j)`` at row ``i * n + j``. The
    ``i == j`` rows are NaN. Quadratic in the cloud, so downsample first;
    this is the descriptor PPF registration hashes."""

    cdef cPPFEstimation[PointXYZ, Normal, PPFSignature]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cPPFEstimation[PointXYZ, Normal, PPFSignature]()
        self.n_input = -1
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def compute(self):
        """Return an ``(n * n, 5)`` float32 array."""
        _need(self.n_input >= 0, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[PPFSignature] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 5)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            view[i, 0] = out[i].f1
            view[i, 1] = out[i].f2
            view[i, 2] = out[i].f3
            view[i, 3] = out[i].f4
            view[i, 4] = out[i].alpha_m
        return result


cdef class PPFRGBEstimation:
    """PPF plus the three colour ratios of each pair
    (pcl::PPFRGBEstimation): ``(n * n, 8)`` rows of ``f1, f2, f3, f4,
    r_ratio, g_ratio, b_ratio, alpha_m`` on a
    :class:`PointCloud_PointXYZRGBA`."""

    cdef cPPFRGBEstimation[PointXYZRGBA, Normal, PPFRGBSignature]* me
    cdef bint has_input
    cdef bint has_normals

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cPPFRGBEstimation[PointXYZRGBA, Normal,
                                        PPFRGBSignature]()
        self.has_input = False
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def compute(self):
        """Return an ``(n * n, 8)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[PPFRGBSignature] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 8)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            view[i, 0] = out[i].f1
            view[i, 1] = out[i].f2
            view[i, 2] = out[i].f3
            view[i, 3] = out[i].f4
            view[i, 4] = out[i].r_ratio
            view[i, 5] = out[i].g_ratio
            view[i, 6] = out[i].b_ratio
            view[i, 7] = out[i].alpha_m
        return result


cdef class CPPFEstimation:
    """Colour Point Pair Features (pcl::CPPFEstimation): ten geometric and
    colour features per ordered pair, ``(n * n, 11)`` rows of
    ``f1 .. f10, alpha_m`` on a :class:`PointCloud_PointXYZRGBA`."""

    cdef cCPPFEstimation[PointXYZRGBA, Normal, CPPFSignature]* me
    cdef bint has_input
    cdef bint has_normals

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cCPPFEstimation[PointXYZRGBA, Normal, CPPFSignature]()
        self.has_input = False
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def compute(self):
        """Return an ``(n * n, 11)`` float32 array."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[CPPFSignature] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 11)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            view[i, 0] = out[i].f1
            view[i, 1] = out[i].f2
            view[i, 2] = out[i].f3
            view[i, 3] = out[i].f4
            view[i, 4] = out[i].f5
            view[i, 5] = out[i].f6
            view[i, 6] = out[i].f7
            view[i, 7] = out[i].f8
            view[i, 8] = out[i].f9
            view[i, 9] = out[i].f10
            view[i, 10] = out[i].alpha_m
        return result


# --- local reference frames ----------------------------------------------

cdef class SHOTLocalReferenceFrameEstimation:
    """The SHOT local reference frame on its own
    (pcl::SHOTLocalReferenceFrameEstimation): three axes per point from
    the distance-weighted covariance of its neighbourhood, signs
    disambiguated by the majority of points. ``(n, 9)``: x, y, z axis."""

    cdef cSHOTLRF[PointXYZ, ReferenceFrame]* me
    cdef bint has_input
    cdef bint has_radius

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSHOTLRF[PointXYZ, ReferenceFrame]()
        self.has_input = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def compute(self):
        """Return an ``(n, 9)`` float32 array of row-major axes."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[ReferenceFrame] out
        with nogil:
            self.me.compute(out)
        return _frames(out)


cdef class SHOTLocalReferenceFrameEstimationOMP:
    """`SHOTLocalReferenceFrameEstimation` over several threads."""

    cdef cSHOTLRFOMP[PointXYZ, ReferenceFrame]* me
    cdef bint has_input
    cdef bint has_radius

    def __cinit__(self, PointCloud pc=None, unsigned int threads=0):
        self.me = new cSHOTLRFOMP[PointXYZ, ReferenceFrame]()
        self.me.setNumberOfThreads(threads)
        self.has_input = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, 9)`` float32 array of row-major axes."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[ReferenceFrame] out
        with nogil:
            self.me.compute(out)
        return _frames(out)


cdef class BOARDLocalReferenceFrameEstimation:
    """BOARD local reference frames
    (pcl::BOARDLocalReferenceFrameEstimation, Petrelli and Di Stefano):
    the normal as z, and as x the direction to the neighbour whose normal
    deviates most from it, with hole detection for points on the edge of
    a partial view. ``(n, 9)``."""

    cdef cBOARDLRF[PointXYZ, Normal, ReferenceFrame]* me
    cdef bint has_input
    cdef bint has_normals
    cdef bint has_radius

    def __cinit__(self, PointCloud pc=None):
        self.me = new cBOARDLRF[PointXYZ, Normal, ReferenceFrame]()
        self.has_input = False
        self.has_normals = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_TangentRadius(self, float radius):
        """Radius for the x-axis search (defaults to the search radius)."""
        self.me.setTangentRadius(radius)

    def get_TangentRadius(self):
        return self.me.getTangentRadius()

    def set_FindHoles(self, bint find_holes):
        self.me.setFindHoles(find_holes)

    def get_FindHoles(self):
        return self.me.getFindHoles()

    def set_MarginThresh(self, float threshold):
        self.me.setMarginThresh(threshold)

    def get_MarginThresh(self):
        return self.me.getMarginThresh()

    def set_CheckMarginArraySize(self, int size):
        if size < 1:
            raise ValueError("size must be >= 1")
        self.me.setCheckMarginArraySize(size)

    def get_CheckMarginArraySize(self):
        return self.me.getCheckMarginArraySize()

    def set_HoleSizeProbThresh(self, float threshold):
        self.me.setHoleSizeProbThresh(threshold)

    def get_HoleSizeProbThresh(self):
        return self.me.getHoleSizeProbThresh()

    def set_SteepThresh(self, float threshold):
        self.me.setSteepThresh(threshold)

    def get_SteepThresh(self):
        return self.me.getSteepThresh()

    def compute(self):
        """Return an ``(n, 9)`` float32 array of row-major axes."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[ReferenceFrame] out
        with nogil:
            self.me.compute(out)
        return _frames(out)


cdef class FLARELocalReferenceFrameEstimation:
    """FLARE local reference frames
    (pcl::FLARELocalReferenceFrameEstimation, Petrelli and Di Stefano):
    z from the normals within the search radius, x towards the point
    farthest from the tangent plane within `tangent_radius` (which must be
    set). ``(n, 9)``; `get_SignedDistancesFromHighestPoints()` then gives
    that farthest point's signed distance per frame. Where a frame could
    not be built its row is NaN and its distance is the largest float32
    (PCL's marker)."""

    cdef cFLARELRF[PointXYZ, Normal, ReferenceFrame]* me
    cdef bint has_input
    cdef bint has_normals
    cdef bint has_radius
    cdef bint has_tangent

    def __cinit__(self, PointCloud pc=None):
        self.me = new cFLARELRF[PointXYZ, Normal, ReferenceFrame]()
        self.has_input = False
        self.has_normals = False
        self.has_radius = False
        self.has_tangent = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_TangentRadius(self, float radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setTangentRadius(radius)
        self.has_tangent = True

    def get_TangentRadius(self):
        return self.me.getTangentRadius()

    def set_MarginThresh(self, float threshold):
        self.me.setMarginThresh(threshold)

    def get_MarginThresh(self):
        return self.me.getMarginThresh()

    def set_MinNeighboursForNormalAxis(self, int count):
        self.me.setMinNeighboursForNormalAxis(count)

    def get_MinNeighboursForNormalAxis(self):
        return self.me.getMinNeighboursForNormalAxis()

    def set_MinNeighboursForTangentAxis(self, int count):
        self.me.setMinNeighboursForTangentAxis(count)

    def get_MinNeighboursForTangentAxis(self):
        return self.me.getMinNeighboursForTangentAxis()

    def compute(self):
        """Return an ``(n, 9)`` float32 array of row-major axes."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        _need(self.has_tangent, "set_TangentRadius(...) is required first: "
              "PCL logs and returns nothing without it")
        cdef cPointCloud[ReferenceFrame] out
        with nogil:
            self.me.compute(out)
        return _frames(out)

    def get_SignedDistancesFromHighestPoints(self):
        import numpy as np
        cdef const vector[float]* d = &self.me.getSignedDistancesFromHighestPoints()
        result = np.empty(d.size(), dtype=np.float32)
        cdef float[::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> d.size()):
            view[i] = deref(d)[i]
        return result


# --- intensity descriptors -----------------------------------------------

cdef class IntensityGradientEstimation:
    """The intensity gradient at each point of a
    :class:`PointCloud_PointXYZI`, projected onto its tangent plane
    (pcl::IntensityGradientEstimation). ``(n, 3)``; the input RIFT
    needs."""

    cdef cIntensityGradientEstimation[PointXYZI, Normal, IntensityGradient]* me
    cdef bint has_input
    cdef bint has_normals

    def __cinit__(self, PointCloud_PointXYZI pc=None, unsigned int threads=0):
        self.me = new cIntensityGradientEstimation[PointXYZI, Normal,
                                                   IntensityGradient]()
        self.me.setNumberOfThreads(threads)
        self.has_input = False
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZI pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def compute(self):
        """Return an ``(n, 3)`` float32 array of gradients."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[IntensityGradient] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 3)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            view[i, 0] = out[i].gradient_x
            view[i, 1] = out[i].gradient_y
            view[i, 2] = out[i].gradient_z
        return result


cdef class IntensitySpinEstimation:
    """Intensity spin images (pcl::IntensitySpinEstimation, Lazebnik et
    al.): per point, a 2-D histogram of neighbour distance against
    intensity, ``(n, 20)``.

    PCL fixes the output at 20 values but lets the bin counts be set; a
    product above 20 writes past the end of every histogram. So
    `set_NrDistanceBins` / `set_NrIntensityBins` are only accepted when
    their product is 20 (4 x 5 by default)."""

    cdef cIntensitySpinEstimation[PointXYZI, Histogram20]* me
    cdef bint has_input
    cdef bint has_radius
    cdef int distance_bins
    cdef int intensity_bins

    def __cinit__(self, PointCloud_PointXYZI pc=None):
        self.me = new cIntensitySpinEstimation[PointXYZI, Histogram20]()
        self.distance_bins = self.me.getNrDistanceBins()
        self.intensity_bins = self.me.getNrIntensityBins()
        self.has_input = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZI pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_Bins(self, int distance_bins, int intensity_bins):
        """Distance x intensity bins; the product must be 20."""
        if distance_bins < 1 or intensity_bins < 1 or \
                distance_bins * intensity_bins != 20:
            raise ValueError(
                "distance_bins * intensity_bins must be 20 (the fixed "
                "output size), got %d x %d" % (distance_bins, intensity_bins))
        self.me.setNrDistanceBins(distance_bins)
        self.me.setNrIntensityBins(intensity_bins)
        self.distance_bins = distance_bins
        self.intensity_bins = intensity_bins

    def get_NrDistanceBins(self):
        return self.me.getNrDistanceBins()

    def get_NrIntensityBins(self):
        return self.me.getNrIntensityBins()

    def set_SmoothingBandwith(self, float sigma):
        self.me.setSmoothingBandwith(sigma)

    def get_SmoothingBandwith(self):
        return self.me.getSmoothingBandwith()

    def compute(self):
        """Return an ``(n, 20)`` float32 array (distance-major)."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[Histogram20] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 20)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].histogram[0], 20)
        return result


cdef class RIFTEstimation:
    """Rotation Invariant Feature Transform (pcl::RIFTEstimation,
    Lazebnik et al.): per point, histograms of intensity-gradient
    orientation over concentric rings, ``(n, 32)``. Needs the gradients
    from :class:`IntensityGradientEstimation`.

    As with the spin images, the output size is fixed (32) while the bin
    counts are settable, so `set_Bins` only takes a product of 32 (4
    distance x 8 gradient bins by default)."""

    cdef cRIFTEstimation[PointXYZI, IntensityGradient, Histogram32]* me
    cdef Py_ssize_t n_input
    cdef bint has_gradient
    cdef bint has_radius

    def __cinit__(self, PointCloud_PointXYZI pc=None):
        self.me = new cRIFTEstimation[PointXYZI, IntensityGradient,
                                      Histogram32]()
        self.n_input = -1
        self.has_gradient = False
        self.has_radius = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZI pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_InputGradient(self, gradient):
        """The ``(n, 3)`` gradients `IntensityGradientEstimation` returned
        for the same cloud."""
        import numpy as np
        arr = np.ascontiguousarray(gradient, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("gradient must be an (n, 3) array")
        if self.n_input >= 0 and arr.shape[0] != self.n_input:
            raise ValueError("gradient has %d rows, the cloud %d points"
                             % (arr.shape[0], self.n_input))
        cdef shared_ptr[cPointCloud[IntensityGradient]] cloud
        cloud.reset(new cPointCloud[IntensityGradient]())
        cdef float[:, ::1] view = arr
        cdef Py_ssize_t i
        cdef IntensityGradient g
        for i in range(arr.shape[0]):
            g.gradient_x = view[i, 0]
            g.gradient_y = view[i, 1]
            g.gradient_z = view[i, 2]
            cloud.get().push_back(g)
        self.me.setInputGradient(cloud)
        self.has_gradient = True

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def set_Bins(self, int distance_bins, int gradient_bins):
        """Distance x gradient-orientation bins; the product must be 32."""
        if distance_bins < 1 or gradient_bins < 1 or \
                distance_bins * gradient_bins != 32:
            raise ValueError(
                "distance_bins * gradient_bins must be 32 (the fixed output "
                "size), got %d x %d" % (distance_bins, gradient_bins))
        self.me.setNrDistanceBins(distance_bins)
        self.me.setNrGradientBins(gradient_bins)

    def get_NrDistanceBins(self):
        return self.me.getNrDistanceBins()

    def get_NrGradientBins(self):
        return self.me.getNrGradientBins()

    def compute(self):
        """Return an ``(n, 32)`` float32 array."""
        _need(self.n_input >= 0, "set_InputCloud(...) is required first")
        _need(self.has_gradient, "set_InputGradient(...) is required first")
        _need(self.has_radius, "set_RadiusSearch(...) is required first")
        cdef cPointCloud[Histogram32] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 32)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].histogram[0], 32)
        return result


# --- normals ---------------------------------------------------------------

cdef class LinearLeastSquaresNormalEstimation:
    """Normals of an organized cloud from a least-squares plane over each
    pixel's depth neighbourhood (pcl::LinearLeastSquaresNormalEstimation)
    -- the alternative to `IntegralImageNormalEstimation`. Returns a
    :class:`PointCloud_Normal` on the same pixel grid; pixels without a
    valid neighbourhood get NaN."""

    cdef cLinearLeastSquaresNormalEstimation[PointXYZ, Normal]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cLinearLeastSquaresNormalEstimation[PointXYZ, Normal]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if pc.ptr().height < 2:
            raise ValueError(
                "LinearLeastSquaresNormalEstimation needs an organized cloud "
                "(height > 1); build one with PointCloud.from_organized_array")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_NormalSmoothingSize(self, float size):
        """Half-size of the depth window, in pixels (default 9)."""
        if size <= 0:
            raise ValueError("size must be positive")
        self.me.setNormalSmoothingSize(size)

    def set_DepthDependentSmoothing(self, bint enable):
        """Grow the window with depth, as sensor noise does."""
        self.me.setDepthDependentSmoothing(enable)

    def set_MaxDepthChangeFactor(self, float factor):
        """Depth jump (per unit of depth) treated as an edge."""
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.me.setMaxDepthChangeFactor(factor)

    def compute(self):
        """Return a :class:`PointCloud_Normal` on the input's grid."""
        _need(self.has_input, "set_InputCloud(...) is required first")
        cdef shared_ptr[cPointCloud[Normal]] holder
        holder.reset(new cPointCloud[Normal]())
        cdef cPointCloud[Normal]* out = holder.get()
        with nogil:
            self.me.compute(deref(out))
        return wrap_normal_cloud(holder)


# --- organized edges ---------------------------------------------------------

cdef object _edge_result(cPointCloud[Label]& labels,
                         vector[PointIndices]& indices):
    """``(labels (h, w) uint32, [indices per edge type])``."""
    import numpy as np
    flat = np.empty(labels.size(), dtype=np.uint32)
    cdef uint32_t[::1] view = flat
    cdef Py_ssize_t i, j
    for i in range(<Py_ssize_t> labels.size()):
        view[i] = labels[i].label
    lists = []
    for i in range(<Py_ssize_t> indices.size()):
        arr = np.empty(indices[i].indices.size(), dtype=np.int32)
        for j in range(<Py_ssize_t> indices[i].indices.size()):
            arr[j] = indices[i].indices[j]
        lists.append(arr)
    return flat.reshape(labels.height, labels.width), lists




cdef object _organized_or_raise(Py_ssize_t height, str name):
    if height < 2:
        raise ValueError(
            "%s needs an organized cloud (height > 1)" % name)


cdef class OrganizedEdgeBase:
    """Depth edges of an organized cloud
    (pcl::OrganizedEdgeBase): NaN boundaries, and both sides of each depth
    discontinuity -- OCCLUDING is the near side, OCCLUDED the far one.

    `compute()` returns ``(labels, indices)``: a ``(height, width)``
    uint32 array whose bits are the ``pcl.EDGELABEL_*`` types found at
    each pixel, and one int32 index array per type, in bit order
    (NAN_BOUNDARY, OCCLUDING, OCCLUDED, HIGH_CURVATURE, RGB_CANNY).
    `set_EdgeType` takes an OR of the ``pcl.EDGELABEL_*`` bits to look
    for.
    """

    cdef cOrganizedEdgeBase[PointXYZ, Label]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cOrganizedEdgeBase[PointXYZ, Label]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        _organized_or_raise(pc.ptr().height, "OrganizedEdgeBase")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_DepthDisconThreshold(self, float threshold):
        """Depth jump, relative to depth, that counts as an edge."""
        self.me.setDepthDisconThreshold(threshold)

    def get_DepthDisconThreshold(self):
        return self.me.getDepthDisconThreshold()

    def set_MaxSearchNeighbors(self, int count):
        self.me.setMaxSearchNeighbors(count)

    def get_MaxSearchNeighbors(self):
        return self.me.getMaxSearchNeighbors()

    def set_EdgeType(self, int edge_types):
        self.me.setEdgeType(edge_types)

    def get_EdgeType(self):
        return self.me.getEdgeType()

    def compute(self):
        _need(self.has_input, "set_InputCloud(...) is required first")
        cdef cPointCloud[Label] labels
        cdef vector[PointIndices] indices
        with nogil:
            self.me.compute(labels, indices)
        return _edge_result(labels, indices)


cdef class OrganizedEdgeFromNormals:
    """Depth edges plus HIGH_CURVATURE edges from the normals
    (pcl::OrganizedEdgeFromNormals): Canny on the normal field, with
    `set_HCCannyLowThreshold` / `High` as its hysteresis thresholds.

    `compute()` returns ``(labels, indices)``: a ``(height, width)``
    uint32 array whose bits are the ``pcl.EDGELABEL_*`` types found at
    each pixel, and one int32 index array per type, in bit order
    (NAN_BOUNDARY, OCCLUDING, OCCLUDED, HIGH_CURVATURE, RGB_CANNY).
    `set_EdgeType` takes an OR of the ``pcl.EDGELABEL_*`` bits to look
    for.
    """

    cdef cOrganizedEdgeFromNormals[PointXYZ, Normal, Label]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cOrganizedEdgeFromNormals[PointXYZ, Normal, Label]()
        self.n_input = -1
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        _organized_or_raise(pc.ptr().height, "OrganizedEdgeFromNormals")
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_DepthDisconThreshold(self, float threshold):
        self.me.setDepthDisconThreshold(threshold)

    def set_MaxSearchNeighbors(self, int count):
        self.me.setMaxSearchNeighbors(count)

    def set_EdgeType(self, int edge_types):
        self.me.setEdgeType(edge_types)

    def get_EdgeType(self):
        return self.me.getEdgeType()

    def set_HCCannyLowThreshold(self, float threshold):
        self.me.setHCCannyLowThreshold(threshold)

    def set_HCCannyHighThreshold(self, float threshold):
        self.me.setHCCannyHighThreshold(threshold)

    def compute(self):
        _need(self.n_input >= 0, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[Label] labels
        cdef vector[PointIndices] indices
        with nogil:
            self.me.compute(labels, indices)
        return _edge_result(labels, indices)


cdef class OrganizedEdgeFromRGB:
    """Depth edges plus RGB_CANNY edges from the colour image
    (pcl::OrganizedEdgeFromRGB), on a :class:`PointCloud_PointXYZRGBA`.

    `compute()` returns ``(labels, indices)``: a ``(height, width)``
    uint32 array whose bits are the ``pcl.EDGELABEL_*`` types found at
    each pixel, and one int32 index array per type, in bit order
    (NAN_BOUNDARY, OCCLUDING, OCCLUDED, HIGH_CURVATURE, RGB_CANNY).
    `set_EdgeType` takes an OR of the ``pcl.EDGELABEL_*`` bits to look
    for.
    """

    cdef cOrganizedEdgeFromRGB[PointXYZRGBA, Label]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cOrganizedEdgeFromRGB[PointXYZRGBA, Label]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_or_raise(pc.ptr().height, "OrganizedEdgeFromRGB")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_DepthDisconThreshold(self, float threshold):
        self.me.setDepthDisconThreshold(threshold)

    def set_MaxSearchNeighbors(self, int count):
        self.me.setMaxSearchNeighbors(count)

    def set_EdgeType(self, int edge_types):
        self.me.setEdgeType(edge_types)

    def get_EdgeType(self):
        return self.me.getEdgeType()

    def set_RGBCannyLowThreshold(self, float threshold):
        self.me.setRGBCannyLowThreshold(threshold)

    def set_RGBCannyHighThreshold(self, float threshold):
        self.me.setRGBCannyHighThreshold(threshold)

    def compute(self):
        _need(self.has_input, "set_InputCloud(...) is required first")
        cdef cPointCloud[Label] labels
        cdef vector[PointIndices] indices
        with nogil:
            self.me.compute(labels, indices)
        return _edge_result(labels, indices)


cdef class OrganizedEdgeFromRGBNormals:
    """All five edge types at once
    (pcl::OrganizedEdgeFromRGBNormals): depth, high-curvature and colour
    edges of a :class:`PointCloud_PointXYZRGBA` with its normals.

    `compute()` returns ``(labels, indices)``: a ``(height, width)``
    uint32 array whose bits are the ``pcl.EDGELABEL_*`` types found at
    each pixel, and one int32 index array per type, in bit order
    (NAN_BOUNDARY, OCCLUDING, OCCLUDED, HIGH_CURVATURE, RGB_CANNY).
    `set_EdgeType` takes an OR of the ``pcl.EDGELABEL_*`` bits to look
    for.
    """

    cdef cOrganizedEdgeFromRGBNormals[PointXYZRGBA, Normal, Label]* me
    cdef bint has_input
    cdef bint has_normals

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cOrganizedEdgeFromRGBNormals[PointXYZRGBA, Normal,
                                                   Label]()
        self.has_input = False
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        _organized_or_raise(pc.ptr().height, "OrganizedEdgeFromRGBNormals")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_DepthDisconThreshold(self, float threshold):
        self.me.setDepthDisconThreshold(threshold)

    def set_MaxSearchNeighbors(self, int count):
        self.me.setMaxSearchNeighbors(count)

    def set_EdgeType(self, int edge_types):
        self.me.setEdgeType(edge_types)

    def get_EdgeType(self):
        return self.me.getEdgeType()

    def set_HCCannyLowThreshold(self, float threshold):
        self.me.setHCCannyLowThreshold(threshold)

    def set_HCCannyHighThreshold(self, float threshold):
        self.me.setHCCannyHighThreshold(threshold)

    def set_RGBCannyLowThreshold(self, float threshold):
        self.me.setRGBCannyLowThreshold(threshold)

    def set_RGBCannyHighThreshold(self, float threshold):
        self.me.setRGBCannyHighThreshold(threshold)

    def compute(self):
        _need(self.has_input, "set_InputCloud(...) is required first")
        _need(self.has_normals, "set_InputNormals(...) is required first")
        cdef cPointCloud[Label] labels
        cdef vector[PointIndices] indices
        with nogil:
            self.me.compute(labels, indices)
        return _edge_result(labels, indices)


# --- global and mesh descriptors ---------------------------------------------

cdef class GFPFHEstimation:
    """Global FPFH (pcl::GFPFHEstimation, Rusu et al.): one 16-bin
    descriptor for a whole cloud whose points carry surface-class labels
    (for instance from a per-point classifier): how often a ray between
    two occupied octree leaves passes through each pair of classes.

        gfpfh = pcl.GFPFHEstimation()
        gfpfh.set_InputCloud(cloud, labels)     # labels: (n,) ints
        gfpfh.set_NumberOfClasses(3)
        descriptor = gfpfh.compute()            # (16,) float32
    """

    cdef cGFPFHEstimation[PointXYZL, PointXYZL, GFPFHSignature16]* me
    cdef shared_ptr[cPointCloud[PointXYZL]] cloud
    cdef bint has_input

    def __cinit__(self):
        self.me = new cGFPFHEstimation[PointXYZL, PointXYZL,
                                       GFPFHSignature16]()
        self.has_input = False

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None, labels):
        """The cloud and one non-negative integer label per point (labels
        are 1-based classes; 0 means unlabelled)."""
        import numpy as np
        arr = np.ascontiguousarray(labels)
        if arr.ndim != 1 or arr.shape[0] != pc.size:
            raise ValueError("labels must be an (n,) array, one per point")
        if pc.size and (arr.min() < 0):
            raise ValueError("labels must be non-negative")
        cdef long long[::1] view = arr.astype(np.int64)
        self.cloud.reset(new cPointCloud[PointXYZL]())
        cdef cPointCloud[PointXYZ]* src = pc.ptr()
        cdef PointXYZL p
        cdef Py_ssize_t i
        for i in range(pc.size):
            p.x = deref(src)[i].x
            p.y = deref(src)[i].y
            p.z = deref(src)[i].z
            p.label = <uint32_t> view[i]
            self.cloud.get().push_back(p)
        self.me.setInputCloud(self.cloud)
        self.me.setInputLabels(self.cloud)
        self.has_input = True

    def set_OctreeLeafSize(self, double size):
        if size <= 0:
            raise ValueError("size must be positive")
        self.me.setOctreeLeafSize(size)

    def get_OctreeLeafSize(self):
        return self.me.getOctreeLeafSize()

    def set_NumberOfClasses(self, uint32_t count):
        if count < 1:
            raise ValueError("count must be >= 1")
        self.me.setNumberOfClasses(count)

    def get_NumberOfClasses(self):
        return self.me.getNumberOfClasses()

    def compute(self):
        """Return the ``(16,)`` float32 descriptor."""
        import numpy as np
        _need(self.has_input, "set_InputCloud(cloud, labels) is required first")
        cdef cPointCloud[GFPFHSignature16] out
        with nogil:
            self.me.compute(out)
        if out.size() != 1:
            raise RuntimeError("GFPFHEstimation produced no descriptor")
        result = np.empty(16, dtype=np.float32)
        cdef Py_ssize_t j
        for j in range(16):
            result[j] = out[0].histogram[j]
        return result


cdef class ROPSEstimation:
    """Rotational Projection Statistics (pcl::ROPSEstimation, Guo et al.):
    the local MESH around a point is rotated about its reference axes and
    projected onto three planes; five statistics of each projection make
    the ``(k, 135)`` descriptor. Needs the triangles as well as the
    points, so it suits a reconstructed mesh:

        vertices, triangles = mesher.reconstruct()
        rops = pcl.ROPSEstimation(cloud, triangles)
        rops.set_SupportRadius(0.05)
        descriptors = rops.compute(indices)

    The output size is fixed at 135 = rotations x 3 axes x 3 planes x 5
    statistics, so the number of rotations is fixed at PCL's default 3;
    `set_NumberOfPartitionBins` is free.
    """

    cdef cROPSEstimation[PointXYZ, Histogram135]* me
    cdef Py_ssize_t n_input
    cdef bint has_support

    def __cinit__(self, PointCloud pc=None, triangles=None):
        self.me = new cROPSEstimation[PointXYZ, Histogram135]()
        self.n_input = -1
        self.has_support = False
        if pc is not None:
            self.set_InputCloud(pc)
        if triangles is not None:
            self.set_Triangles(triangles)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_Triangles(self, triangles):
        """Index triples into the input cloud."""
        cdef vector[Vertices] polygons
        cdef Vertices v
        for triangle in triangles:
            if len(triangle) != 3:
                raise ValueError("ROPS takes triangles, got a polygon of %d"
                                 % len(triangle))
            v.vertices.clear()
            for index in triangle:
                if index < 0 or (self.n_input >= 0 and index >= self.n_input):
                    raise IndexError("triangle index %d is outside the cloud"
                                     % index)
                v.vertices.push_back(<uint32_t> index)
            polygons.push_back(v)
        self.me.setTriangles(polygons)

    def set_SupportRadius(self, float radius):
        """Radius of the local surface each descriptor covers."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setSupportRadius(radius)
        self.me.setRadiusSearch(radius)
        self.has_support = True

    def get_SupportRadius(self):
        return self.me.getSupportRadius()

    def set_NumberOfPartitionBins(self, unsigned int bins):
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.me.setNumberOfPartitionBins(bins)

    def get_NumberOfPartitionBins(self):
        return self.me.getNumberOfPartitionBins()

    def get_NumberOfRotations(self):
        return self.me.getNumberOfRotations()

    def compute(self, indices=None):
        """Describe the points *indices* (all of them by default); returns
        a ``(k, 135)`` float32 array."""
        _need(self.n_input >= 0, "set_InputCloud(...) is required first")
        _need(self.has_support, "set_SupportRadius(...) is required first")
        cdef shared_ptr[vector[int]] chosen
        if indices is not None:
            chosen.reset(new vector[int]())
            for index in indices:
                if index < 0 or index >= self.n_input:
                    raise IndexError("index %d is outside the cloud" % index)
                chosen.get().push_back(index)
            self.me.setIndices(chosen)
        cdef cPointCloud[Histogram135] out
        with nogil:
            self.me.compute(out)
        result = _rows(<Py_ssize_t> out.size(), 135)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out.size()):
            _copy(view, i, &out[i].histogram[0], 135)
        return result


# --- multiscale ----------------------------------------------------------------

cdef class StatisticalMultiscaleInterestRegionExtraction:
    """Interest regions that stand out across scales
    (pcl::StatisticalMultiscaleInterestRegionExtraction, Ranjan and
    Guskov): geodesic neighbourhoods at each scale, kept where their
    statistics are extremal.

        extractor = pcl.StatisticalMultiscaleInterestRegionExtraction(cloud)
        extractor.set_ScalesVector([0.05, 0.1, 0.2])
        regions = extractor.compute_RegionsOfInterest()  # [int32 arrays]
    """

    cdef PointCloud cloud
    cdef vector[float] scales

    def __cinit__(self, PointCloud pc=None):
        self.cloud = pc

    def set_InputCloud(self, PointCloud pc not None):
        self.cloud = pc

    def set_ScalesVector(self, scales):
        values = [float(s) for s in scales]
        if not values or min(values) <= 0:
            raise ValueError("scales must be a non-empty list of positive "
                             "radii")
        self.scales = values

    def get_ScalesVector(self):
        return list(self.scales)

    def compute_RegionsOfInterest(self):
        """Return one int32 index array per region of interest."""
        import numpy as np
        _need(self.cloud is not None, "set_InputCloud(...) is required first")
        _need(self.scales.size() > 0, "set_ScalesVector(...) is required first")
        cdef vector[vector[int]] regions
        with nogil:
            multiscaleInterestRegions(self.cloud.thisptr_shared, self.scales,
                                      regions)
        result = []
        cdef Py_ssize_t i, j
        for i in range(<Py_ssize_t> regions.size()):
            arr = np.empty(regions[i].size(), dtype=np.int32)
            for j in range(<Py_ssize_t> regions[i].size()):
                arr[j] = regions[i][j]
            result.append(arr)
        return result


cdef class MultiscaleFeaturePersistence:
    """The features that stay distinctive across scales
    (pcl::MultiscaleFeaturePersistence, Rusu et al.), over FPFH: the
    descriptor is computed at every radius in `set_ScalesVector`, and a
    point is kept when, at every scale, its descriptor lies more than
    `alpha` standard deviations from the mean. Returns ``(features,
    indices)``: the ``(k, 33)`` FPFH of the persistent points at the last
    scale and their indices into the cloud.
    """

    cdef PointCloud cloud
    cdef PointCloud_Normal normals
    cdef vector[float] scales
    cdef float alpha
    cdef int norm

    def __cinit__(self, PointCloud pc=None, PointCloud_Normal normals=None):
        self.cloud = pc
        self.normals = normals
        self.alpha = 1.0
        self.norm = ft.NORM_L1

    def set_InputCloud(self, PointCloud pc not None):
        self.cloud = pc

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.normals = normals

    def set_ScalesVector(self, scales):
        values = [float(s) for s in scales]
        if not values or min(values) <= 0:
            raise ValueError("scales must be a non-empty list of positive "
                             "radii")
        self.scales = values

    def set_Alpha(self, float alpha):
        """Standard deviations from the mean a feature must keep."""
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def get_Alpha(self):
        return self.alpha

    def set_DistanceMetric(self, int norm):
        """One of the ``pcl.NORM_*`` metrics (default ``NORM_L1``)."""
        if not 0 <= norm <= ft.NORM_HIK:
            raise ValueError("unknown norm %r" % norm)
        self.norm = norm

    def determine_PersistentFeatures(self):
        import numpy as np
        _need(self.cloud is not None, "set_InputCloud(...) is required first")
        _need(self.normals is not None,
              "set_InputNormals(...) is required first")
        _need(self.scales.size() > 0, "set_ScalesVector(...) is required first")
        _check_normals(self.cloud.size, self.normals)
        cdef cPointCloud[FPFHSignature33] features
        cdef vector[int] indices
        with nogil:
            persistentFPFH(self.cloud.thisptr_shared,
                           self.normals.thisptr_shared, self.scales,
                           self.alpha, self.norm, features, indices)
        result = _rows(<Py_ssize_t> features.size(), 33)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> features.size()):
            _copy(view, i, &features[i].histogram[0], 33)
        idx = np.empty(indices.size(), dtype=np.int32)
        for i in range(<Py_ssize_t> indices.size()):
            idx[i] = indices[i]
        return result, idx


# --- functions -------------------------------------------------------------------

def compute_approximate_normals(PointCloud cloud not None, polygons):
    """Per-vertex normals of a mesh, averaged from the faces around each
    vertex (pcl::features::computeApproximateNormals). Returns a
    :class:`PointCloud_Normal`, one normal per vertex."""
    cdef vector[Vertices] faces
    cdef Vertices v
    cdef Py_ssize_t n = cloud.size
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("polygons need at least three vertices")
        v.vertices.clear()
        for index in polygon:
            if index < 0 or index >= n:
                raise IndexError("polygon index %d is outside the cloud"
                                 % index)
            v.vertices.push_back(<uint32_t> index)
        faces.push_back(v)
    cdef shared_ptr[cPointCloud[Normal]] holder
    holder.reset(new cPointCloud[Normal]())
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    with nogil:
        approximateNormals(deref(src), faces, deref(holder.get()))
    return wrap_normal_cloud(holder)


def compute_pair_features(p1, n1, p2, n2):
    """The four PFH pair features of two oriented points
    (pcl::computePairFeatures): three angles and the distance, as
    ``(f1, f2, f3, f4)``. Raises for a degenerate pair (coincident
    points)."""
    cdef float a[3]
    cdef float na[3]
    cdef float b[3]
    cdef float nb[3]
    cdef float out[4]
    for arr, src in ((0, p1), (1, n1), (2, p2), (3, n2)):
        values = [float(x) for x in src]
        if len(values) != 3:
            raise ValueError("points and normals must have three components")
        for k in range(3):
            if arr == 0:
                a[k] = values[k]
            elif arr == 1:
                na[k] = values[k]
            elif arr == 2:
                b[k] = values[k]
            else:
                nb[k] = values[k]
    if not pairFeatures(a, na, b, nb, out):
        raise ValueError("the two points coincide; pair features are "
                         "undefined")
    return out[0], out[1], out[2], out[3]
