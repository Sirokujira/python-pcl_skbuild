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
