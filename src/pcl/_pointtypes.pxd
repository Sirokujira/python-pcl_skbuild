# Hand-written declaration of the typed point-cloud classes.
#
# NOT generated: these are the Cython-level interfaces of cdef classes we
# wrote, not projections of C++ headers, so they do not belong under
# src/pcl/pxd/. Only PointCloud_Normal is reached from another module so
# far (features -> segmentation), but declaring the set keeps them
# usable the same way.
from libcpp.memory cimport shared_ptr

from pcl.pxd.point_types cimport (
    Normal, PointXYZI, PointXYZRGB, PointXYZRGBA)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud


cdef class PointCloud_PointXYZI:
    cdef shared_ptr[cPointCloud[PointXYZI]] thisptr_shared
    cdef cPointCloud[PointXYZI]* ptr(self) except NULL


cdef class PointCloud_PointXYZRGB:
    cdef shared_ptr[cPointCloud[PointXYZRGB]] thisptr_shared
    cdef cPointCloud[PointXYZRGB]* ptr(self) except NULL


cdef class PointCloud_PointXYZRGBA:
    cdef shared_ptr[cPointCloud[PointXYZRGBA]] thisptr_shared
    cdef cPointCloud[PointXYZRGBA]* ptr(self) except NULL


cdef class PointCloud_Normal:
    cdef shared_ptr[cPointCloud[Normal]] thisptr_shared
    cdef cPointCloud[Normal]* ptr(self) except NULL


cdef PointCloud_Normal wrap_normal_cloud(shared_ptr[cPointCloud[Normal]] ptr)
