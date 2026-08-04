# distutils: language = c++
# cython: language_level=3
"""Surface wrappers, named after sirokujira/python-pcl
(pcl/pxi/Surface/).

Reach them through `PointCloud.make_moving_least_squares()`,
`PointCloud.make_ConcaveHull()` and `PointCloud.make_ConvexHull()`.

MovingLeastSquares smooths a cloud by fitting a polynomial to each
neighbourhood. `process()` returns a PointCloud; when
`set_Compute_Normals(True)` was set, `process_with_normals()` returns the
same points plus the estimated normals as a second array — PCL writes
both into one PointNormal cloud, and splitting them keeps PointCloud
meaning one thing.
"""

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointXYZ, PointNormal
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.surface.mls cimport MovingLeastSquares as cMovingLeastSquares
from pcl.pxd.surface.concave_hull cimport ConcaveHull as cConcaveHull
from pcl.pxd.surface.convex_hull cimport ConvexHull as cConvexHull

from pcl._pointcloud cimport PointCloud, wrap_cloud


cdef class MovingLeastSquares:
    """Polynomial surface smoothing (pcl::MovingLeastSquares)."""

    cdef cMovingLeastSquares[PointXYZ, PointNormal]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cMovingLeastSquares[PointXYZ, PointNormal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_search_radius(self, double radius):
        """Neighbourhood size the polynomial is fitted over. Required."""
        self.me.setSearchRadius(radius)

    def get_search_radius(self):
        return self.me.getSearchRadius()

    def set_polynomial_order(self, int order):
        self.me.setPolynomialOrder(order)

    def get_polynomial_order(self):
        return self.me.getPolynomialOrder()

    def set_Compute_Normals(self, bint compute):
        self.me.setComputeNormals(compute)

    def set_polynomial_fit(self, bint fit):
        """python-pcl spelling: order 2 when fitting, 1 when not."""
        self.me.setPolynomialOrder(2 if fit else 1)

    cdef _run(self, cPointCloud[PointNormal]* out):
        cdef cMovingLeastSquares[PointXYZ, PointNormal]* mls = self.me
        with nogil:
            mls.process(deref(out))

    def process(self):
        """Return the smoothed cloud as a :class:`PointCloud`."""
        cdef cPointCloud[PointNormal] out
        cdef Py_ssize_t i, n
        self._run(&out)

        cdef PointCloud result = PointCloud()
        n = <Py_ssize_t> out.size()
        result.resize(n)
        cdef cPointCloud[PointXYZ]* dst = result.ptr()
        for i in range(n):
            deref(dst)[<size_t> i].x = out[<size_t> i].x
            deref(dst)[<size_t> i].y = out[<size_t> i].y
            deref(dst)[<size_t> i].z = out[<size_t> i].z
        return result

    def process_with_normals(self):
        """Return ``(cloud, normals)``; *normals* is an ``(n, 4)`` float32
        array of ``normal_x, normal_y, normal_z, curvature``.

        Only meaningful after ``set_Compute_Normals(True)``.
        """
        import numpy as np
        cdef cPointCloud[PointNormal] out
        cdef Py_ssize_t i, n
        self._run(&out)

        n = <Py_ssize_t> out.size()
        cdef PointCloud result = PointCloud()
        result.resize(n)
        cdef cPointCloud[PointXYZ]* dst = result.ptr()
        normals = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = normals
        for i in range(n):
            deref(dst)[<size_t> i].x = out[<size_t> i].x
            deref(dst)[<size_t> i].y = out[<size_t> i].y
            deref(dst)[<size_t> i].z = out[<size_t> i].z
            view[i, 0] = out[<size_t> i].normal_x
            view[i, 1] = out[<size_t> i].normal_y
            view[i, 2] = out[<size_t> i].normal_z
            view[i, 3] = out[<size_t> i].curvature
        return result, normals


cdef class ConcaveHull:
    """Alpha-shape hull of a cloud (pcl::ConcaveHull)."""

    cdef cConcaveHull[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cConcaveHull[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_Alpha(self, double alpha):
        """Maximum edge length of the hull. Required, and the whole knob:
        larger alpha gives a hull closer to the convex one."""
        self.me.setAlpha(alpha)

    def get_Alpha(self):
        return self.me.getAlpha()

    def set_Dimension(self, int dimension):
        self.me.setDimension(dimension)

    def get_Dimension(self):
        return self.me.getDimension()

    def reconstruct(self):
        cdef PointCloud result = PointCloud()
        cdef cPointCloud[PointXYZ]* out = result.ptr()
        cdef cConcaveHull[PointXYZ]* hull = self.me
        with nogil:
            hull.reconstruct(deref(out))
        return result


cdef class ConvexHull:
    """Convex hull of a cloud (pcl::ConvexHull)."""

    cdef cConvexHull[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cConvexHull[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_Dimension(self, int dimension):
        self.me.setDimension(dimension)

    def get_Dimension(self):
        return self.me.getDimension()

    def set_ComputeAreaVolume(self, bint value):
        """Area and volume are only filled in when this is set first."""
        self.me.setComputeAreaVolume(value)

    def get_TotalArea(self):
        return self.me.getTotalArea()

    def get_TotalVolume(self):
        return self.me.getTotalVolume()

    def reconstruct(self):
        cdef PointCloud result = PointCloud()
        cdef cPointCloud[PointXYZ]* out = result.ptr()
        cdef cConvexHull[PointXYZ]* hull = self.me
        with nogil:
            hull.reconstruct(deref(out))
        return result
