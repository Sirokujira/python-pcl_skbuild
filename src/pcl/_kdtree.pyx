# distutils: language = c++
# cython: language_level=3
"""KdTreeFLANN wrapper, named after sirokujira/python-pcl
(pcl/pxi/KdTree/KdTree_FLANN.pxi).

Reach it through `PointCloud.make_kdtree_flann()`, or construct it with the
cloud to index.

The search methods return whole-cloud results in one call and fill numpy
arrays directly: one crossing of the language boundary per query cloud
rather than one per point (see bench/README.md).
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.kdtree.kdtree_flann cimport KdTreeFLANN as cKdTreeFLANN

from pcl._pointcloud cimport PointCloud


cdef class KdTreeFLANN:
    """FLANN-backed kd-tree over a PointCloud (pcl::KdTreeFLANN)."""

    cdef cKdTreeFLANN[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cKdTreeFLANN[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_epsilon(self, float eps):
        self.me.setEpsilon(eps)

    def get_epsilon(self):
        return self.me.getEpsilon()

    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """k nearest neighbours for every point of *pc*.

        Returns ``(indices, sqr_distances)``, both of shape ``(len(pc), k)``.
        """
        if k < 1:
            raise ValueError("k must be >= 1, got %d" % k)
        import numpy as np

        cdef cPointCloud[PointXYZ]* query = pc.ptr()
        cdef Py_ssize_t n = <Py_ssize_t> query.size()

        ind = np.zeros((n, k), dtype=np.int32)
        sqdist = np.zeros((n, k), dtype=np.float32)
        cdef int[:, ::1] ind_view = ind
        cdef float[:, ::1] sqdist_view = sqdist

        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        cdef Py_ssize_t i, j
        cdef int found

        k_indices.resize(k)
        k_sqr_distances.resize(k)
        for i in range(n):
            with nogil:
                found = self.me.nearestKSearch(
                    deref(query)[<size_t> i], k, k_indices, k_sqr_distances)
            for j in range(found):
                ind_view[i, j] = k_indices[j]
                sqdist_view[i, j] = k_sqr_distances[j]
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud pc not None,
                                   Py_ssize_t index, int k=1):
        """k nearest neighbours of a single point of *pc*.

        Returns ``(indices, sqr_distances)``, both of length ``k``.
        """
        if k < 1:
            raise ValueError("k must be >= 1, got %d" % k)
        cdef Py_ssize_t n = <Py_ssize_t> pc.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        import numpy as np

        cdef cPointCloud[PointXYZ]* query = pc.ptr()
        ind = np.zeros(k, dtype=np.int32)
        sqdist = np.zeros(k, dtype=np.float32)
        cdef int[::1] ind_view = ind
        cdef float[::1] sqdist_view = sqdist

        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        cdef int found
        cdef Py_ssize_t j

        k_indices.resize(k)
        k_sqr_distances.resize(k)
        with nogil:
            found = self.me.nearestKSearch(
                deref(query)[<size_t> index], k, k_indices, k_sqr_distances)
        for j in range(found):
            ind_view[j] = k_indices[j]
            sqdist_view[j] = k_sqr_distances[j]
        return ind, sqdist

    def radius_search_for_cloud(self, PointCloud pc not None, double radius,
                                unsigned int max_nn=0):
        """Neighbours within *radius* for every point of *pc*.

        Returns ``(indices, sqr_distances)`` of shape ``(len(pc), max_nn)``.
        Rows are zero-padded past the number of neighbours actually found,
        so *max_nn* must be given — an unbounded search has no rectangular
        shape to return.
        """
        if max_nn == 0:
            raise ValueError(
                "max_nn must be > 0: the result is a fixed-width array, so "
                "the per-point neighbour count has to be bounded")
        import numpy as np

        cdef cPointCloud[PointXYZ]* query = pc.ptr()
        cdef Py_ssize_t n = <Py_ssize_t> query.size()

        ind = np.zeros((n, max_nn), dtype=np.int32)
        sqdist = np.zeros((n, max_nn), dtype=np.float32)
        cdef int[:, ::1] ind_view = ind
        cdef float[:, ::1] sqdist_view = sqdist

        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        cdef Py_ssize_t i, j
        cdef int found

        k_indices.resize(max_nn)
        k_sqr_distances.resize(max_nn)
        for i in range(n):
            with nogil:
                found = self.me.radiusSearch(
                    deref(query)[<size_t> i], radius, k_indices,
                    k_sqr_distances, max_nn)
            for j in range(found):
                ind_view[i, j] = k_indices[j]
                sqdist_view[i, j] = k_sqr_distances[j]
        return ind, sqdist
