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

from pcl.pxd.point_types cimport PointXYZ, Normal, PointNormal, VFHSignature308
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.features.normal_3d cimport NormalEstimation as cNormalEstimation
from pcl.pxd.features.moment_of_inertia_estimation cimport (
    MomentOfInertiaEstimation as cMomentOfInertiaEstimation)
from pcl.pxd.features.vfh cimport VFHEstimation as cVFHEstimation
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

    def __cinit__(self, PointCloud pc=None):
        self.me = new cIntegralImageNormalEstimation[PointXYZ, Normal]()
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
        return wrap_normal_cloud(holder)


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
