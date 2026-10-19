# distutils: language = c++
# cython: language_level=3
"""Feature wrappers, named after sirokujira/python-pcl
(pcl/pxi/Features/).

Reach them through `PointCloud.make_NormalEstimation()` and
`PointCloud.make_MomentOfInertiaEstimation()`.

`NormalEstimation` gives its result two ways. `compute()` returns an
``(n, 4)`` numpy array — ``normal_x, normal_y, normal_z, curvature`` —
which is what most callers want and what python-pcl's own
`PointCloud_Normal` gets converted into anyway. `compute_cloud()`
returns a :class:`PointCloud_Normal`, the form `SegmentationNormal`
takes, so normal-based segmentation never has to round-trip through
Python.

MomentOfInertiaEstimation's Eigen-typed getters go through
pcl/compat/eigen_results.h, which flattens them into float buffers — a
mirror header may not name Eigen, and the values are numpy-bound anyway.
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport (
    PointXYZ, Normal, PointNormal, VFHSignature308, FPFHSignature33, SHOT352)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.features.normal_3d cimport NormalEstimation as cNormalEstimation
from pcl.pxd.features.fpfh cimport FPFHEstimation as cFPFHEstimation
from pcl.pxd.features.shot cimport SHOTEstimation as cSHOTEstimation
from pcl.pxd.features.moment_of_inertia_estimation cimport (
    MomentOfInertiaEstimation as cMomentOfInertiaEstimation)
from pcl.pxd.features.vfh cimport VFHEstimation as cVFHEstimation
from pcl.pxd.point_types cimport Boundary, ESFSignature640
from pcl.pxd.point_types cimport (
    PrincipalCurvatures, PFHSignature125, PrincipalRadiiRSD,
    GASDSignature512, GASDSignature984, GRSDSignature21, MomentInvariants,
    PointXYZRGB)
from pcl.pxd.compat.feature_types cimport (
    Histogram153, Histogram90, setCRHCentroid)
from pcl.pxd.features.principal_curvatures cimport (
    PrincipalCurvaturesEstimation as cPrincipalCurvaturesEstimation)
from pcl.pxd.features.normal_3d_omp cimport (
    NormalEstimationOMP as cNormalEstimationOMP)
from pcl.pxd.features.pfh cimport PFHEstimation as cPFHEstimation
from pcl.pxd.features.rsd cimport RSDEstimation as cRSDEstimation
from pcl.pxd.features.spin_image cimport (
    SpinImageEstimation as cSpinImageEstimation)
from pcl.pxd.features.gasd cimport (
    GASDEstimation as cGASDEstimation,
    GASDColorEstimation as cGASDColorEstimation,
    HistogramInterpolationMethod)
cimport pcl.pxd.features.gasd as gasd_types
from pcl.pxd.features.grsd cimport GRSDEstimation as cGRSDEstimation
from pcl.pxd.features.our_cvfh cimport (
    OURCVFHEstimation as cOURCVFHEstimation)
from pcl.pxd.features.crh cimport CRHEstimation as cCRHEstimation
from pcl.pxd.features.moment_invariants cimport (
    MomentInvariantsEstimation as cMomentInvariantsEstimation)
from pcl._pointtypes cimport PointCloud_PointXYZRGB
from pcl.pxd.features.boundary cimport (
    BoundaryEstimation as cBoundaryEstimation)
from pcl.pxd.features.esf cimport ESFEstimation as cESFEstimation
from pcl.pxd.features.cvfh cimport CVFHEstimation as cCVFHEstimation
from libc.stdint cimport uint8_t
from pcl.pxd.features.don cimport (
    DifferenceOfNormalsEstimation as cDifferenceOfNormalsEstimation)
from pcl.pxd.features.integral_image_normal cimport (
    IntegralImageNormalEstimation as cIntegralImageNormalEstimation)
from pcl.pxd.compat.organized_args cimport (
    IINE_AVERAGE_3D_GRADIENT, IINE_AVERAGE_DEPTH_CHANGE,
    IINE_COVARIANCE_MATRIX, IINE_SIMPLE_3D_GRADIENT,
    setNormalEstimationMethod)
from pcl.pxd.compat.eigen_results cimport (
    axisAlignedBoundingBox, eigenValues, eigenVectors, massCenter,
    orientedBoundingBox)

from libcpp.memory cimport shared_ptr

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_Normal, wrap_normal_cloud


cdef class NormalEstimation:
    """Surface normals from each point's neighbourhood
    (pcl::NormalEstimation).

    Set exactly one of `set_KSearch` (k nearest neighbours) or
    `set_RadiusSearch` (all neighbours within a radius) before computing.
    """

    cdef cNormalEstimation[PointXYZ, Normal]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cNormalEstimation[PointXYZ, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def get_KSearch(self):
        return self.me.getKSearch()

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def compute_cloud(self):
        """Return the normals as a :class:`PointCloud_Normal`.

        This is the form `SegmentationNormal.set_InputNormals()` takes;
        `compute()` gives the same values as an array.
        """
        cdef shared_ptr[cPointCloud[Normal]] holder
        holder.reset(new cPointCloud[Normal]())
        cdef cPointCloud[Normal]* out = holder.get()
        with nogil:
            self.me.compute(deref(out))
        return wrap_normal_cloud(holder)

    def compute(self):
        """Return an ``(n, 4)`` float32 array:
        ``normal_x, normal_y, normal_z, curvature``."""
        import numpy as np
        cdef cPointCloud[Normal] out
        with nogil:
            self.me.compute(out)

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Normal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.normal_x
            view[i, 1] = p.normal_y
            view[i, 2] = p.normal_z
            view[i, 3] = p.curvature
        return result


cdef class MomentOfInertiaEstimation:
    """Inertia/eccentricity descriptors and bounding boxes
    (pcl::MomentOfInertiaEstimation).

    Call `compute()` once, then read whichever results you need.
    """

    cdef cMomentOfInertiaEstimation[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cMomentOfInertiaEstimation[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_AngleStep(self, float step):
        self.me.setAngleStep(step)

    def get_AngleStep(self):
        return self.me.getAngleStep()

    def set_NormalizePointMassFlag(self, bint normalize):
        self.me.setNormalizePointMassFlag(normalize)

    def set_PointMass(self, float mass):
        self.me.setPointMass(mass)

    def compute(self):
        cdef cMomentOfInertiaEstimation[PointXYZ]* est = self.me
        with nogil:
            est.compute()
        return self

    def get_MomentOfInertia(self):
        cdef vector[float] values
        self.me.getMomentOfInertia(values)
        return [values[i] for i in range(values.size())]

    def get_Eccentricity(self):
        cdef vector[float] values
        self.me.getEccentricity(values)
        return [values[i] for i in range(values.size())]

    def get_MassCenter(self):
        """Centre of mass as a length-3 float32 array."""
        import numpy as np
        cdef float buf[3]
        massCenter(deref(self.me), buf)
        return np.array([buf[0], buf[1], buf[2]], dtype=np.float32)

    def get_AABB(self):
        """Axis-aligned bounding box as ``(min_xyz, max_xyz)``."""
        import numpy as np
        cdef float buf[6]
        axisAlignedBoundingBox(deref(self.me), buf)
        return (np.array([buf[0], buf[1], buf[2]], dtype=np.float32),
                np.array([buf[3], buf[4], buf[5]], dtype=np.float32))

    def get_OBB(self):
        """Oriented bounding box as
        ``(min_xyz, max_xyz, position_xyz, rotation_3x3)``."""
        import numpy as np
        cdef float buf[18]
        cdef int i
        orientedBoundingBox(deref(self.me), buf)
        rotation = np.empty((3, 3), dtype=np.float32)
        for i in range(9):
            rotation[i // 3, i % 3] = buf[9 + i]
        return (np.array([buf[0], buf[1], buf[2]], dtype=np.float32),
                np.array([buf[3], buf[4], buf[5]], dtype=np.float32),
                np.array([buf[6], buf[7], buf[8]], dtype=np.float32),
                rotation)

    def get_EigenValues(self):
        """``(major, middle, minor)``."""
        cdef float buf[3]
        eigenValues(deref(self.me), buf)
        return (buf[0], buf[1], buf[2])

    def get_EigenVectors(self):
        """``(major, middle, minor)``, each a length-3 float32 array."""
        import numpy as np
        cdef float buf[9]
        eigenVectors(deref(self.me), buf)
        return tuple(
            np.array([buf[k], buf[k + 1], buf[k + 2]], dtype=np.float32)
            for k in (0, 3, 6)
        )


cdef class VFHEstimation:
    """Viewpoint Feature Histogram (pcl::VFHEstimation).

    One 308-bin descriptor for the whole cloud, used to recognise an
    object independently of its pose. Needs normals:

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        vfh = cloud.make_VFHEstimation()
        vfh.set_InputNormals(normals.compute_cloud())
        histogram = vfh.compute()
    """

    cdef cVFHEstimation[PointXYZ, Normal, VFHSignature308]* me
    # PCL segfaults rather than complaining when normals are missing, so
    # the wrapper has to know whether they were set.
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cVFHEstimation[PointXYZ, Normal, VFHSignature308]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def set_NormalizeBins(self, bint normalize):
        self.me.setNormalizeBins(normalize)

    def set_NormalizeDistance(self, bint normalize):
        self.me.setNormalizeDistance(normalize)

    def set_FillSizeComponent(self, bint fill_size):
        self.me.setFillSizeComponent(fill_size)

    def compute(self):
        """Return the descriptor as a length-308 float32 array.

        VFH describes the whole cloud, so PCL's output holds one point;
        the histogram inside it is what callers want.

        Raises if no normals were set: PCL segfaults in that case rather
        than reporting the error — reproduced in plain C++ against
        1.14.0, so it is not a binding bug and nothing can catch it.
        """
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "VFHEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud()). "
                "PCL segfaults instead of reporting this.")

        cdef cPointCloud[VFHSignature308] out
        with nogil:
            self.me.compute(out)

        if out.size() == 0:
            raise RuntimeError("VFH produced no descriptor")

        result = np.empty(308, dtype=np.float32)
        cdef float[::1] view = result
        cdef VFHSignature308* p = &out[0]
        cdef Py_ssize_t i
        for i in range(308):
            view[i] = p.histogram[i]
        return result


cdef class BoundaryEstimation:
    """Flag the points on the edge of a surface (pcl::BoundaryEstimation).

    Each point's neighbours are projected onto its tangent plane; a gap
    wider than `set_AngleThreshold` (radians, default pi/2) between
    consecutive neighbours marks the point as a boundary. Needs normals
    and a neighbourhood size:

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        edges = cloud.make_BoundaryEstimation()
        edges.set_InputNormals(normals.compute_cloud())
        edges.set_KSearch(20)
        on_edge = edges.compute()        # bool array, one per point
    """

    cdef cBoundaryEstimation[PointXYZ, Normal, Boundary]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cBoundaryEstimation[PointXYZ, Normal, Boundary]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def get_KSearch(self):
        return self.me.getKSearch()

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def set_AngleThreshold(self, float angle):
        """Widest angular gap (radians) still counted as surrounded."""
        self.me.setAngleThreshold(angle)

    def get_AngleThreshold(self):
        return self.me.getAngleThreshold()

    def compute(self):
        """Return a bool array, True where the point lies on a boundary."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "BoundaryEstimation needs normals: call set_InputNormals() "
                "with a PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[Boundary] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        flags = np.empty(n, dtype=np.uint8)
        cdef uint8_t[::1] view = flags
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = out[<size_t> i].boundary_point
        return flags.astype(np.bool_)


cdef class ESFEstimation:
    """Ensemble of Shape Functions (pcl::ESFEstimation).

    One 640-bin descriptor for the whole cloud, built from the distances,
    angles and areas of randomly sampled point pairs and triples. Needs
    neither normals nor a neighbourhood size, so it is the cheapest
    global descriptor here:

        histogram = cloud.make_ESFEstimation().compute()
    """

    cdef cESFEstimation[PointXYZ, ESFSignature640]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cESFEstimation[PointXYZ, ESFSignature640]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def compute(self):
        """Return the descriptor as a length-640 float32 array.

        The point samples are drawn at random, so two runs on the same
        cloud differ slightly.
        """
        import numpy as np
        cdef cPointCloud[ESFSignature640] out
        with nogil:
            self.me.compute(out)
        if out.size() == 0:
            raise RuntimeError(
                "ESF produced no descriptor (is the input cloud empty?)")
        result = np.empty(640, dtype=np.float32)
        cdef float[::1] view = result
        cdef ESFSignature640* p = &out[0]
        cdef Py_ssize_t i
        for i in range(640):
            view[i] = p.histogram[i]
        return result


cdef class CVFHEstimation:
    """Clustered Viewpoint Feature Histogram (pcl::CVFHEstimation).

    VFH computed once per smooth region: regions are grown over points
    whose normals agree within `set_EPSAngleThreshold` and whose
    curvature stays below `set_CurvatureThreshold`, within
    `set_ClusterTolerance` of each other, so a partial view yields one
    308-bin descriptor per region. When no region reaches
    `set_MinPoints`, PCL falls back to a single VFH of the whole cloud.
    Needs normals, and `set_RadiusNormals` / `set_ClusterTolerance` in
    the cloud's own units (the defaults assume metres):

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        cvfh = cloud.make_CVFHEstimation()
        cvfh.set_InputNormals(normals.compute_cloud())
        cvfh.set_ClusterTolerance(0.2)       # a few point spacings
        cvfh.set_RadiusNormals(0.2)
        histograms = cvfh.compute()         # (regions, 308)
    """

    cdef cCVFHEstimation[PointXYZ, Normal, VFHSignature308]* me
    # Like VFH, PCL dereferences the normals without checking them.
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cCVFHEstimation[PointXYZ, Normal, VFHSignature308]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def set_RadiusNormals(self, float radius):
        """Radius within which CVFH RE-ESTIMATES normals on the smooth
        candidates before clustering (PCL's default 0.03 is metre-scale).
        Below the cloud's point spacing every point ends up alone, no
        region forms, and the whole-cloud fallback quietly takes over —
        so set this to a few point spacings."""
        self.me.setRadiusNormals(radius)

    def set_ClusterTolerance(self, float distance):
        """How far apart two points may be and still share a region, in
        the cloud's own units (PCL's default 0.015 suits metre-scale
        sensor data)."""
        self.me.setClusterTolerance(distance)

    def set_EPSAngleThreshold(self, float angle):
        """Largest angle (radians) between neighbouring normals within
        one region."""
        self.me.setEPSAngleThreshold(angle)

    def set_CurvatureThreshold(self, float curvature):
        """Points above this curvature are left out of every region."""
        self.me.setCurvatureThreshold(curvature)

    def set_MinPoints(self, size_t count):
        """Smallest region that gets its own descriptor."""
        self.me.setMinPoints(count)

    def set_NormalizeBins(self, bint normalize):
        self.me.setNormalizeBins(normalize)

    def compute(self):
        """Return a ``(regions, 308)`` float32 array, one row per smooth
        region (a single row when no region was large enough)."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "CVFHEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud()). "
                "PCL dereferences them unchecked.")
        cdef cPointCloud[VFHSignature308] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        if n == 0:
            raise RuntimeError("CVFH produced no descriptor")
        result = np.empty((n, 308), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef VFHSignature308* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(308):
                view[i, j] = p.histogram[j]
        return result


cdef class PrincipalCurvaturesEstimation:
    """Principal curvatures of each point's neighbourhood
    (pcl::PrincipalCurvaturesEstimation).

    From the normals around a point: the direction of greatest curvature
    and the two curvature magnitudes `pc1 >= pc2`. Needs normals and
    `set_KSearch` / `set_RadiusSearch`.
    """

    cdef cPrincipalCurvaturesEstimation[PointXYZ, Normal, PrincipalCurvatures]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cPrincipalCurvaturesEstimation[PointXYZ, Normal, PrincipalCurvatures]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def compute(self):
        """Return an ``(n, 5)`` float32 array, one row per input point:
        principal_curvature_x, principal_curvature_y, principal_curvature_z, pc1, pc2."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "PrincipalCurvaturesEstimation needs normals: call set_InputNormals() with a "
                "PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[PrincipalCurvatures] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 5), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PrincipalCurvatures* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.principal_curvature_x
            view[i, 1] = p.principal_curvature_y
            view[i, 2] = p.principal_curvature_z
            view[i, 3] = p.pc1
            view[i, 4] = p.pc2
        return result


cdef class PFHEstimation:
    """Point Feature Histograms (pcl::PFHEstimation).

    The exact 125-bin ancestor of FPFH: every pair in a point's
    neighbourhood contributes, so it costs O(k^2) per point. Needs
    normals and `set_KSearch` / `set_RadiusSearch`.
    """

    cdef cPFHEstimation[PointXYZ, Normal, PFHSignature125]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cPFHEstimation[PointXYZ, Normal, PFHSignature125]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_MaximumCacheSize(self, unsigned int size):
        self.me.setMaximumCacheSize(size)

    def get_MaximumCacheSize(self):
        return self.me.getMaximumCacheSize()

    def set_UseInternalCache(self, bint use_cache):
        self.me.setUseInternalCache(use_cache)

    def get_UseInternalCache(self):
        return self.me.getUseInternalCache()

    def compute(self):
        """Return an ``(n, 125)`` float32 array, one histogram per
        input point."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "PFHEstimation needs normals: call set_InputNormals() with a "
                "PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[PFHSignature125] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 125), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PFHSignature125* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(125):
                view[i, j] = p.histogram[j]
        return result


cdef class RSDEstimation:
    """Radius-based Surface Descriptor (pcl::RSDEstimation).

    The smallest and largest surface radius around each point, from the
    distance/angle relation to its neighbours. Radius search only --
    PCL's setKSearch is a no-op that prints an error. Radii above
    `set_PlaneRadius` (default 0.2) are clamped as "planar", so raise it
    for large or coarse clouds.
    """

    cdef cRSDEstimation[PointXYZ, Normal, PrincipalRadiiRSD]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cRSDEstimation[PointXYZ, Normal, PrincipalRadiiRSD]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_NrSubdivisions(self, int count):
        self.me.setNrSubdivisions(count)

    def get_NrSubdivisions(self):
        return self.me.getNrSubdivisions()

    def set_PlaneRadius(self, double radius):
        """Radius above which a surface counts as a plane; results are
        clamped to it."""
        self.me.setPlaneRadius(radius)

    def get_PlaneRadius(self):
        return self.me.getPlaneRadius()

    def compute(self):
        """Return an ``(n, 2)`` float32 array, one row per input point:
        r_min, r_max."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "RSDEstimation needs normals: call set_InputNormals() with a "
                "PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[PrincipalRadiiRSD] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 2), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PrincipalRadiiRSD* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.r_min
            view[i, 1] = p.r_max
        return result


cdef class SpinImageEstimation:
    """Spin images (pcl::SpinImageEstimation).

    Each point's neighbourhood as a 2-D histogram over (radial distance,
    height along the normal) -- an 8-wide image, 153 bins. Needs normals
    and `set_KSearch` / `set_RadiusSearch`.
    """

    cdef cSpinImageEstimation[PointXYZ, Normal, Histogram153]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSpinImageEstimation[PointXYZ, Normal, Histogram153]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_ImageWidth(self, unsigned int width):
        """Bins per side; the output is sized for the default 8."""
        if width != 8:
            raise ValueError(
                "only the default image width 8 (153 bins) is supported")
        self.me.setImageWidth(width)

    def set_SupportAngle(self, double support_angle_cos):
        """Cosine of the largest angle between a neighbour's normal and
        the point's for it to count (0 = everything)."""
        self.me.setSupportAngle(support_angle_cos)

    def set_MinPointCountInNeighbourhood(self, unsigned int count):
        self.me.setMinPointCountInNeighbourhood(count)

    def set_AngularDomain(self, bint is_angular):
        self.me.setAngularDomain(is_angular)

    def set_RadialStructure(self, bint is_radial):
        self.me.setRadialStructure(is_radial)

    def compute(self):
        """Return an ``(n, 153)`` float32 array, one histogram per
        input point."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "SpinImageEstimation needs normals: call set_InputNormals() with a "
                "PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[Histogram153] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 153), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Histogram153* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(153):
                view[i, j] = p.histogram[j]
        return result


cdef class NormalEstimationOMP:
    """`NormalEstimation` parallelised with OpenMP
    (pcl::NormalEstimationOMP). Same calls, plus `set_NumberOfThreads`
    (0, the default, lets OpenMP decide).
    """

    cdef cNormalEstimationOMP[PointXYZ, Normal]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cNormalEstimationOMP[PointXYZ, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def get_KSearch(self):
        return self.me.getKSearch()

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def compute_cloud(self):
        """Return the normals as a :class:`PointCloud_Normal`."""
        cdef shared_ptr[cPointCloud[Normal]] holder
        holder.reset(new cPointCloud[Normal]())
        cdef cPointCloud[Normal]* out = holder.get()
        with nogil:
            self.me.compute(deref(out))
        return wrap_normal_cloud(holder)

    def compute(self):
        """Return an ``(n, 4)`` float32 array:
        ``normal_x, normal_y, normal_z, curvature``."""
        import numpy as np
        cdef cPointCloud[Normal] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Normal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.normal_x
            view[i, 1] = p.normal_y
            view[i, 2] = p.normal_z
            view[i, 3] = p.curvature
        return result


# GASD histogram interpolation, re-exported from the header so the
# values cannot drift.
INTERP_NONE = gasd_types.INTERP_NONE
INTERP_TRILINEAR = gasd_types.INTERP_TRILINEAR
INTERP_QUADRILINEAR = gasd_types.INTERP_QUADRILINEAR


cdef _check_interp(int method):
    if method not in (gasd_types.INTERP_NONE, gasd_types.INTERP_TRILINEAR,
                      gasd_types.INTERP_QUADRILINEAR):
        raise ValueError(
            "unknown interpolation %d: use pcl.INTERP_NONE, "
            "pcl.INTERP_TRILINEAR or pcl.INTERP_QUADRILINEAR" % method)


cdef class GASDEstimation:
    """Globally Aligned Spatial Distribution (pcl::GASDEstimation).

    One 512-bin descriptor for the whole cloud: the cloud is aligned to
    its own reference frame (centroid and principal axes) and a 3-D grid
    of occupancy histograms is taken over it. Needs neither normals nor a
    neighbourhood size:

        histogram = cloud.make_GASDEstimation().compute()
    """

    cdef cGASDEstimation[PointXYZ, GASDSignature512]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cGASDEstimation[PointXYZ, GASDSignature512]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_ShapeHalfGridSize(self, size_t size):
        """Half the grid size per axis (default 4: an 8x8x8 grid). The
        output is sized for the defaults; other values change the
        descriptor length PCL writes into the fixed 512 bins."""
        self.me.setShapeHalfGridSize(size)

    def set_ShapeHistsSize(self, size_t size):
        self.me.setShapeHistsSize(size)

    def set_ShapeHistsInterpMethod(self, int method):
        _check_interp(method)
        self.me.setShapeHistsInterpMethod(<HistogramInterpolationMethod> method)

    def compute(self):
        """Return the descriptor as a length-512 float32 array."""
        import numpy as np
        cdef cPointCloud[GASDSignature512] out
        with nogil:
            self.me.compute(out)
        if out.size() == 0:
            raise RuntimeError(
                "GASD produced no descriptor (is the input cloud empty?)")
        result = np.empty(512, dtype=np.float32)
        cdef float[::1] view = result
        cdef GASDSignature512* p = &out[0]
        cdef Py_ssize_t i
        for i in range(512):
            view[i] = p.histogram[i]
        return result


cdef class GASDColorEstimation:
    """GASD with colour (pcl::GASDColorEstimation) on a
    :class:`PointCloud_PointXYZRGB`: the 512 shape bins followed by 472
    colour bins, 984 in all.
    """

    cdef cGASDColorEstimation[PointXYZRGB, GASDSignature984]* me

    def __cinit__(self, PointCloud_PointXYZRGB pc=None):
        self.me = new cGASDColorEstimation[PointXYZRGB, GASDSignature984]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_ShapeHalfGridSize(self, size_t size):
        self.me.setShapeHalfGridSize(size)

    def set_ShapeHistsSize(self, size_t size):
        self.me.setShapeHistsSize(size)

    def set_ShapeHistsInterpMethod(self, int method):
        _check_interp(method)
        self.me.setShapeHistsInterpMethod(<HistogramInterpolationMethod> method)

    def set_ColorHalfGridSize(self, size_t size):
        self.me.setColorHalfGridSize(size)

    def set_ColorHistsSize(self, size_t size):
        self.me.setColorHistsSize(size)

    def set_ColorHistsInterpMethod(self, int method):
        _check_interp(method)
        self.me.setColorHistsInterpMethod(<HistogramInterpolationMethod> method)

    def compute(self):
        """Return the descriptor as a length-984 float32 array."""
        import numpy as np
        cdef cPointCloud[GASDSignature984] out
        with nogil:
            self.me.compute(out)
        if out.size() == 0:
            raise RuntimeError(
                "GASD produced no descriptor (is the input cloud empty?)")
        result = np.empty(984, dtype=np.float32)
        cdef float[::1] view = result
        cdef GASDSignature984* p = &out[0]
        cdef Py_ssize_t i
        for i in range(984):
            view[i] = p.histogram[i]
        return result


cdef class GRSDEstimation:
    """Global Radius-based Surface Descriptor (pcl::GRSDEstimation).

    One 21-bin descriptor for the whole cloud from how its RSD surface
    types (plane, edge, corner, ...) neighbour each other. Needs normals
    and `set_RadiusSearch`, which also sets the voxel size.
    """

    cdef cGRSDEstimation[PointXYZ, Normal, GRSDSignature21]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cGRSDEstimation[PointXYZ, Normal, GRSDSignature21]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def set_NrSubdivisions(self, int count):
        self.me.setNrSubdivisions(count)

    def set_PlaneRadius(self, double radius):
        self.me.setPlaneRadius(radius)

    def compute(self):
        """Return the descriptor as a length-21 float32 array."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "GRSDEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[GRSDSignature21] out
        with nogil:
            self.me.compute(out)
        if out.size() == 0:
            raise RuntimeError(
                "GRSD produced no descriptor (set_RadiusSearch missing?)")
        result = np.empty(21, dtype=np.float32)
        cdef float[::1] view = result
        cdef GRSDSignature21* p = &out[0]
        cdef Py_ssize_t i
        for i in range(21):
            view[i] = p.histogram[i]
        return result


cdef class OURCVFHEstimation:
    """Oriented, Unique and Repeatable CVFH (pcl::OURCVFHEstimation).

    `CVFHEstimation` plus a repeatable reference frame per smooth region,
    so each 308-bin descriptor also fixes the object's roll. Same
    settings and the same `set_RadiusNormals` trap as CVFH; `compute()`
    returns ``(regions, 308)``.
    """

    cdef cOURCVFHEstimation[PointXYZ, Normal, VFHSignature308]* me
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cOURCVFHEstimation[PointXYZ, Normal, VFHSignature308]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def set_RadiusNormals(self, float radius):
        """Radius for the normal re-estimation before clustering: keep
        it above the cloud's point spacing (see CVFHEstimation)."""
        self.me.setRadiusNormals(radius)

    def set_ClusterTolerance(self, float distance):
        self.me.setClusterTolerance(distance)

    def set_EPSAngleThreshold(self, float angle):
        self.me.setEPSAngleThreshold(angle)

    def set_CurvatureThreshold(self, float curvature):
        self.me.setCurvatureThreshold(curvature)

    def set_MinPoints(self, size_t count):
        self.me.setMinPoints(count)

    def set_NormalizeBins(self, bint normalize):
        self.me.setNormalizeBins(normalize)

    def set_RefineClusters(self, float factor):
        """Fraction (0..1) of a region's points that must agree before
        the region is refined; 1 disables."""
        self.me.setRefineClusters(factor)

    def set_AxisRatio(self, float ratio):
        self.me.setAxisRatio(ratio)

    def set_MinAxisValue(self, float value):
        self.me.setMinAxisValue(value)

    def compute(self):
        """Return a ``(regions, 308)`` float32 array."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "OURCVFHEstimation needs normals: call set_InputNormals() "
                "with a PointCloud_Normal (NormalEstimation.compute_cloud())")
        cdef cPointCloud[VFHSignature308] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        if n == 0:
            raise RuntimeError("OUR-CVFH produced no descriptor")
        result = np.empty((n, 308), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef VFHSignature308* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(308):
                view[i, j] = p.histogram[j]
        return result


cdef class CRHEstimation:
    """Camera Roll Histogram (pcl::CRHEstimation).

    A 90-bin histogram of the cloud's roll about the line from the
    viewpoint to its centroid -- the one degree of freedom CVFH leaves
    open. PCL needs the centroid handed in; it is computed from the input
    cloud unless `set_Centroid` was called. Needs normals.
    """

    cdef cCRHEstimation[PointXYZ, Normal, Histogram90]* me
    cdef bint has_normals
    cdef bint has_centroid
    cdef PointCloud _input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cCRHEstimation[PointXYZ, Normal, Histogram90]()
        self.has_normals = False
        self.has_centroid = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def set_Centroid(self, float x, float y, float z):
        """Centroid the roll is measured around (default: the input
        cloud's mean)."""
        setCRHCentroid(deref(self.me), x, y, z)
        self.has_centroid = True

    def compute(self):
        """Return the histogram as a length-90 float32 array."""
        import numpy as np
        if not self.has_normals:
            raise RuntimeError(
                "CRHEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud())")
        if self._input is None or self._input.size == 0:
            raise RuntimeError("no input cloud, or an empty one")
        if not self.has_centroid:
            centroid = self._input.to_array()[:, :3].mean(axis=0)
            setCRHCentroid(deref(self.me), centroid[0], centroid[1],
                           centroid[2])
        cdef cPointCloud[Histogram90] out
        with nogil:
            self.me.compute(out)
        if out.size() == 0:
            raise RuntimeError("CRH produced no descriptor")
        result = np.empty(90, dtype=np.float32)
        cdef float[::1] view = result
        cdef Histogram90* p = &out[0]
        cdef Py_ssize_t i
        for i in range(90):
            view[i] = p.histogram[i]
        return result


cdef class MomentInvariantsEstimation:
    """The three moment invariants of each point's neighbourhood
    (pcl::MomentInvariantsEstimation). Needs `set_KSearch` /
    `set_RadiusSearch`; returns ``(n, 3)``: j1, j2, j3.
    """

    cdef cMomentInvariantsEstimation[PointXYZ, MomentInvariants]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cMomentInvariantsEstimation[PointXYZ, MomentInvariants]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_KSearch(self, int k):
        self.me.setKSearch(k)

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)

    def compute(self):
        """Return an ``(n, 3)`` float32 array: j1, j2, j3 per point."""
        import numpy as np
        cdef cPointCloud[MomentInvariants] out
        with nogil:
            self.me.compute(out)
        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 3), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef MomentInvariants* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.j1
            view[i, 1] = p.j2
            view[i, 2] = p.j3
        return result


# IntegralImageNormalEstimation::NormalEstimationMethod, re-exported from
# the header through the shim so the values cannot drift.
COVARIANCE_MATRIX = IINE_COVARIANCE_MATRIX
AVERAGE_3D_GRADIENT = IINE_AVERAGE_3D_GRADIENT
AVERAGE_DEPTH_CHANGE = IINE_AVERAGE_DEPTH_CHANGE
SIMPLE_3D_GRADIENT = IINE_SIMPLE_3D_GRADIENT


cdef class IntegralImageNormalEstimation:
    """Normals straight from a depth image
    (pcl::IntegralImageNormalEstimation).

    Much faster than `NormalEstimation`, but it needs an ORGANIZED cloud
    — build one with `PointCloud.from_organized_array()`. On an
    unorganized cloud PCL returns nothing, so the wrapper says so rather
    than handing back an empty array.
    """

    cdef cIntegralImageNormalEstimation[PointXYZ, Normal]* me
    cdef unsigned int width
    cdef unsigned int height
    cdef bint computed

    def __cinit__(self, PointCloud pc=None):
        self.me = new cIntegralImageNormalEstimation[PointXYZ, Normal]()
        self.computed = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if not pc.is_organized:
            raise ValueError(
                "IntegralImageNormalEstimation needs an organized cloud "
                "(height > 1); build one with "
                "PointCloud.from_organized_array()")
        self.me.setInputCloud(pc.thisptr_shared)
        self.width = pc.width
        self.height = pc.height
        self.computed = False

    def set_NormalEstimationMethod(self, int method):
        """One of `pcl.COVARIANCE_MATRIX`, `AVERAGE_3D_GRADIENT`,
        `AVERAGE_DEPTH_CHANGE`, `SIMPLE_3D_GRADIENT`."""
        setNormalEstimationMethod(deref(self.me), method)

    def set_MaxDepthChangeFactor(self, float factor):
        self.me.setMaxDepthChangeFactor(factor)

    def set_NormalSmoothingSize(self, float size):
        self.me.setNormalSmoothingSize(size)

    def set_DepthDependentSmoothing(self, bint enable):
        self.me.setDepthDependentSmoothing(enable)

    def set_ViewPoint(self, float x, float y, float z):
        self.me.setViewPoint(x, y, z)

    def compute(self):
        """Return an ``(n, 4)`` float32 array:
        ``normal_x, normal_y, normal_z, curvature``."""
        import numpy as np
        cdef cPointCloud[Normal] out
        with nogil:
            self.me.compute(out)
        self.computed = True

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Normal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.normal_x
            view[i, 1] = p.normal_y
            view[i, 2] = p.normal_z
            view[i, 3] = p.curvature
        return result

    def compute_cloud(self):
        """Return the normals as a :class:`PointCloud_Normal`."""
        cdef shared_ptr[cPointCloud[Normal]] holder
        holder.reset(new cPointCloud[Normal]())
        cdef cPointCloud[Normal]* out = holder.get()
        with nogil:
            self.me.compute(deref(out))
        self.computed = True
        return wrap_normal_cloud(holder)

    def get_DistanceMap(self):
        """Per pixel, the distance in pixels to the nearest depth change
        or invalid pixel, as a ``(height, width)`` float32 array -- what
        `EdgeAwarePlaneComparator` takes. Filled by the last
        `compute()` / `compute_cloud()`."""
        import numpy as np
        cdef float* data = self.me.getDistanceMap()
        if not self.computed or data == NULL:
            raise RuntimeError("call compute() or compute_cloud() first")
        result = np.empty((self.height, self.width), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t r, c
        for r in range(self.height):
            for c in range(self.width):
                view[r, c] = data[r * self.width + c]
        return result


cdef class DifferenceOfNormalsEstimation:
    """Scale-based feature isolation (pcl::DifferenceOfNormalsEstimation).

    The difference between normals estimated at two radii is large
    exactly where the surface has structure at that scale, so
    thresholding it isolates features of a chosen size:

        small = cloud.make_NormalEstimation(); small.set_RadiusSearch(0.05)
        large = cloud.make_NormalEstimation(); large.set_RadiusSearch(0.5)
        don = cloud.make_DifferenceOfNormalsEstimation()
        don.set_NormalScaleSmall(small.compute_cloud())
        don.set_NormalScaleLarge(large.compute_cloud())
        magnitude = don.compute()[:, 3]

    The two radii must differ, and the larger is the scale of the
    features you keep.
    """

    cdef cDifferenceOfNormalsEstimation[PointXYZ, Normal, PointNormal]* me
    cdef Py_ssize_t n_input
    cdef Py_ssize_t n_small
    cdef Py_ssize_t n_large

    def __cinit__(self, PointCloud pc=None):
        self.me = new cDifferenceOfNormalsEstimation[
            PointXYZ, Normal, PointNormal]()
        self.n_input = -1
        self.n_small = -1
        self.n_large = -1
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_NormalScaleSmall(self, PointCloud_Normal normals not None):
        self.me.setNormalScaleSmall(normals.thisptr_shared)
        self.n_small = normals.size

    def set_NormalScaleLarge(self, PointCloud_Normal normals not None):
        self.me.setNormalScaleLarge(normals.thisptr_shared)
        self.n_large = normals.size

    def compute(self):
        """Return an ``(n, 4)`` float32 array: the difference vector and
        its magnitude in the fourth column."""
        import numpy as np
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud() is required before compute()")
        if self.n_small < 0 or self.n_large < 0:
            raise RuntimeError(
                "both set_NormalScaleSmall() and set_NormalScaleLarge() "
                "are required before compute()")
        if self.n_small != self.n_input or self.n_large != self.n_input:
            raise ValueError(
                "both normal clouds must have one normal per input point: "
                "input %d, small %d, large %d"
                % (self.n_input, self.n_small, self.n_large))

        cdef cPointCloud[PointNormal] out
        if not self.me.initCompute():
            raise RuntimeError("DifferenceOfNormalsEstimation setup failed")
        # PCL never sizes this for us. DoN makes Feature::compute()
        # private, and that is the method that would have resized the
        # output; computeFeature() writes output[i] for every input point
        # regardless. Calling it on an empty cloud segfaults — verified in
        # plain C++ against PCL 1.14.
        out.resize(<size_t> self.n_input)
        with nogil:
            self.me.computeFeature(out)

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef PointNormal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &out[<size_t> i]
            view[i, 0] = p.normal_x
            view[i, 1] = p.normal_y
            view[i, 2] = p.normal_z
            view[i, 3] = p.curvature
        return result


cdef class FPFHEstimation:
    """Fast Point Feature Histograms (pcl::FPFHEstimation).

    One 33-bin descriptor per point — the workhorse of PCL's
    registration and recognition tutorials. Matching two clouds'
    descriptors yields the (model, scene) correspondences that
    GeometricConsistencyGrouping / Hough3DGrouping turn into poses:

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        fpfh = cloud.make_FPFHEstimation()
        fpfh.set_InputNormals(normals.compute_cloud())
        fpfh.set_RadiusSearch(0.05)
        descriptors = fpfh.compute()          # (n, 33) float32
    """

    cdef cFPFHEstimation[PointXYZ, Normal, FPFHSignature33]* me
    cdef Py_ssize_t n_input
    cdef Py_ssize_t n_normals
    cdef bint has_search

    def __cinit__(self, PointCloud pc=None):
        self.me = new cFPFHEstimation[PointXYZ, Normal, FPFHSignature33]()
        self.n_input = -1
        self.n_normals = -1
        self.has_search = False
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
        self.n_normals = normals.size

    def set_KSearch(self, int k):
        self.me.setKSearch(k)
        self.has_search = True

    def get_KSearch(self):
        return self.me.getKSearch()

    def set_RadiusSearch(self, double radius):
        self.me.setRadiusSearch(radius)
        self.has_search = True

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def compute(self):
        """Return an ``(n, 33)`` float32 array, one histogram per point.

        The guards exist because PCL's own failure mode for each of these
        is an error print followed by an EMPTY output cloud — a silent
        no-op by the time it reaches Python.
        """
        import numpy as np
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud() is required before compute()")
        if self.n_normals < 0:
            raise RuntimeError(
                "FPFHEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud())")
        if self.n_normals != self.n_input:
            raise ValueError(
                "normals must have one entry per input point: "
                "input %d, normals %d" % (self.n_input, self.n_normals))
        if not self.has_search:
            raise RuntimeError(
                "set_KSearch() or set_RadiusSearch() is required before "
                "compute()")

        cdef cPointCloud[FPFHSignature33] out
        with nogil:
            self.me.compute(out)

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 33), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef FPFHSignature33* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(33):
                view[i, j] = p.histogram[j]
        return result


cdef class SHOTEstimation:
    """Signature of Histograms of Orientations (pcl::SHOTEstimation).

    The 352-bin descriptor PCL's correspondence-grouping tutorial
    matches. Unlike FPFH it requires a RADIUS — PCL rejects a k-nearest
    setup for it — so that is the only search setting offered:

        shot = cloud.make_SHOTEstimation()
        shot.set_InputNormals(normals)
        shot.set_RadiusSearch(0.05)
        descriptors = shot.compute()          # (n, 352) float32

    A point whose neighbourhood is too sparse for a stable reference
    frame gets a NaN descriptor — that is PCL reporting "no descriptor
    here", and callers filter with ``np.isfinite`` before matching.
    """

    cdef cSHOTEstimation[PointXYZ, Normal, SHOT352]* me
    cdef Py_ssize_t n_input
    cdef Py_ssize_t n_normals
    cdef bint has_radius

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSHOTEstimation[PointXYZ, Normal, SHOT352]()
        self.n_input = -1
        self.n_normals = -1
        self.has_radius = False
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
        self.n_normals = normals.size

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be > 0, got %r" % radius)
        self.me.setRadiusSearch(radius)
        self.has_radius = True

    def get_RadiusSearch(self):
        return self.me.getRadiusSearch()

    def set_LRFRadius(self, float radius):
        """Support radius of the local reference frames, when it should
        differ from the descriptor radius."""
        self.me.setLRFRadius(radius)

    def get_LRFRadius(self):
        return self.me.getLRFRadius()

    def compute(self):
        """Return an ``(n, 352)`` float32 array, one descriptor per point.

        Guarded like FPFH: PCL's own failure mode for a missing input is
        an error print and an empty output.
        """
        import numpy as np
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud() is required before compute()")
        if self.n_normals < 0:
            raise RuntimeError(
                "SHOTEstimation needs normals: call set_InputNormals() with "
                "a PointCloud_Normal (NormalEstimation.compute_cloud())")
        if self.n_normals != self.n_input:
            raise ValueError(
                "normals must have one entry per input point: "
                "input %d, normals %d" % (self.n_input, self.n_normals))
        if not self.has_radius:
            raise RuntimeError(
                "set_RadiusSearch() is required before compute(): SHOT "
                "rejects a k-nearest setup")

        cdef cPointCloud[SHOT352] out
        with nogil:
            self.me.compute(out)

        cdef Py_ssize_t n = <Py_ssize_t> out.size()
        result = np.empty((n, 352), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef SHOT352* p
        cdef Py_ssize_t i, j
        for i in range(n):
            p = &out[<size_t> i]
            for j in range(352):
                view[i, j] = p.descriptor[j]
        return result
