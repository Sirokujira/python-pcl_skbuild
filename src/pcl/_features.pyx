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

from pcl.pxd.point_types cimport PointXYZ, Normal
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.features.normal_3d cimport NormalEstimation as cNormalEstimation
from pcl.pxd.features.moment_of_inertia_estimation cimport (
    MomentOfInertiaEstimation as cMomentOfInertiaEstimation)
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
