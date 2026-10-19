# distutils: language = c++
# cython: language_level=3
"""PCL classes built WITHOUT OpenMP.

`pcl::HarrisKeypoint6D` is header-only, and its OpenMP loops use
`default(none)` without listing the variables they read; OpenMP 4
compilers (GCC 9 and later) reject that outright, and PCL never
instantiates the class itself, so nothing upstream notices (same on PCL
master). Compiled without OpenMP the pragmas are ignored and the code is
correct, only serial -- so this extension is built with the
PCL_WITHOUT_OPENMP target property (see CMakeLists.txt) and holds just
the classes that need it.
"""

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointXYZI, PointXYZRGB
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.keypoints.harris_6d cimport HarrisKeypoint6D as cHarrisKeypoint6D

from pcl._pointtypes cimport PointCloud_PointXYZRGB


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


cdef class HarrisKeypoint6D:
    """Harris corners in geometry AND intensity (pcl::HarrisKeypoint6D):
    the covariance of normals and intensity gradients around each point
    of a :class:`PointCloud_PointXYZRGB` (luminance as the intensity).
    The normals are computed internally over `radius`. `compute()`
    returns ``(n, 4)``: x, y, z, response.

    Single-threaded: this module is compiled without OpenMP (see the
    module docstring), so there is no `set_NumberOfThreads`.
    """

    cdef cHarrisKeypoint6D[PointXYZRGB, PointXYZI]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGB pc=None, float radius=0.01,
                  float threshold=0.0):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me = new cHarrisKeypoint6D[PointXYZRGB, PointXYZI](radius,
                                                               threshold)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_Radius(self, float radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.me.setRadius(radius)

    def set_Threshold(self, float threshold):
        self.me.setThreshold(threshold)

    def set_NonMaxSupression(self, bint suppress):
        self.me.setNonMaxSupression(suppress)

    def set_Refine(self, bint refine):
        self.me.setRefine(refine)

    def compute(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef cPointCloud[PointXYZI] out
        with nogil:
            self.me.compute(out)
        return _xyzi_rows(out)
