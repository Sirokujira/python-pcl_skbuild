# Hand-written declaration of the PointCloud extension type.
#
# NOT generated — this is the Cython-level interface of a cdef class we
# wrote, not a projection of a C++ header, so it does not belong under
# src/pcl/pxd/. It exists so the other extension modules (_filters,
# _kdtree, _segmentation) can cimport PointCloud and reach the underlying
# pcl::PointCloud without going through Python.
from libcpp.memory cimport shared_ptr

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud


cdef class PointCloud:
    cdef shared_ptr[cPointCloud[PointXYZ]] thisptr_shared

    cdef cPointCloud[PointXYZ]* ptr(self) except NULL

    # Every cdef method has to be declared here once the type has a pxd,
    # even the internal ones.
    cdef Py_ssize_t _normalize_index(self, index) except -1
