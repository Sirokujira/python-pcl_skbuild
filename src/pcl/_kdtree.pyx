# distutils: language = c++
# cython: language_level=3
"""Neighbour search: KdTreeFLANN, named after sirokujira/python-pcl
(pcl/pxi/KdTree/KdTree_FLANN.pxi), and OrganizedNeighbor
(pcl::search::OrganizedNeighbor), which python-pcl never had.

Reach KdTreeFLANN through `PointCloud.make_kdtree_flann()`, or construct
either one with the cloud to index.

The search methods return whole-cloud results in one call and fill numpy
arrays directly: one crossing of the language boundary per query cloud
rather than one per point (see bench/README.md). Both classes share the
loops below through a fused type, so they answer in exactly the same
shapes.
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.kdtree.kdtree_flann cimport KdTreeFLANN as cKdTreeFLANN
from pcl.pxd.search.organized cimport OrganizedNeighbor as cOrganizedNeighbor
from pcl.pxd.compat.search_args cimport organizedCameraMatrix

from pcl._pointcloud cimport PointCloud


ctypedef fused Searcher:
    cKdTreeFLANN[PointXYZ]
    cOrganizedNeighbor[PointXYZ]


cdef object _k_search_cloud(Searcher* tree, PointCloud pc, int k):
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
            found = tree.nearestKSearch(
                deref(query)[<size_t> i], k, k_indices, k_sqr_distances)
        for j in range(found):
            ind_view[i, j] = k_indices[j]
            sqdist_view[i, j] = k_sqr_distances[j]
    return ind, sqdist


cdef object _k_search_point(Searcher* tree, PointCloud pc, Py_ssize_t index,
                            int k):
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
        found = tree.nearestKSearch(
            deref(query)[<size_t> index], k, k_indices, k_sqr_distances)
    for j in range(found):
        ind_view[j] = k_indices[j]
        sqdist_view[j] = k_sqr_distances[j]
    return ind, sqdist


cdef object _radius_search_cloud(Searcher* tree, PointCloud pc, double radius,
                                 unsigned int max_nn):
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
            found = tree.radiusSearch(
                deref(query)[<size_t> i], radius, k_indices,
                k_sqr_distances, max_nn)
        for j in range(found):
            ind_view[i, j] = k_indices[j]
            sqdist_view[i, j] = k_sqr_distances[j]
    return ind, sqdist


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
        return _k_search_cloud(self.me, pc, k)

    def nearest_k_search_for_point(self, PointCloud pc not None,
                                   Py_ssize_t index, int k=1):
        """k nearest neighbours of a single point of *pc*.

        Returns ``(indices, sqr_distances)``, both of length ``k``.
        """
        return _k_search_point(self.me, pc, index, k)

    def radius_search_for_cloud(self, PointCloud pc not None, double radius,
                                unsigned int max_nn=0):
        """Neighbours within *radius* for every point of *pc*.

        Returns ``(indices, sqr_distances)`` of shape ``(len(pc), max_nn)``.
        Rows are zero-padded past the number of neighbours actually found,
        so *max_nn* must be given — an unbounded search has no rectangular
        shape to return.
        """
        return _radius_search_cloud(self.me, pc, radius, max_nn)


cdef class OrganizedNeighbor:
    """Neighbour search over an organized cloud from a projective camera
    (pcl::search::OrganizedNeighbor).

    No tree is built: the camera matrix is estimated from the cloud, and
    each query is projected into the image and answered from the pixels
    around it, which is far cheaper than a kd-tree on a full RGB-D frame.
    `set_InputCloud` raises when the cloud is not organized or does not
    fit a pinhole camera (PCL would only log that and then answer
    wrongly). Same search methods and result shapes as
    :class:`KdTreeFLANN`.
    """

    cdef cOrganizedNeighbor[PointXYZ]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None, bint sorted_results=False,
                  float eps=1e-4, unsigned int pyramid_level=5):
        self.me = new cOrganizedNeighbor[PointXYZ](sorted_results, eps,
                                                   pyramid_level)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if pc.ptr().height < 2:
            raise ValueError(
                "OrganizedNeighbor needs an organized cloud (height > 1); "
                "build one with PointCloud.from_organized_array")
        if not self.me.setInputCloud(pc.thisptr_shared):
            self.has_input = False
            raise ValueError(
                "the cloud does not fit a pinhole camera (PCL could not "
                "estimate a projection matrix from it); use KdTreeFLANN")
        self.has_input = True

    cdef cOrganizedNeighbor[PointXYZ]* _ready(self) except NULL:
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        return self.me

    def get_CameraMatrix(self):
        """The 3x3 camera matrix estimated from the input cloud."""
        import numpy as np
        cdef float buf[9]
        organizedCameraMatrix(deref(self._ready()), buf)
        return np.array([buf[i] for i in range(9)],
                        dtype=np.float32).reshape(3, 3)

    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """k nearest neighbours for every point of *pc*; see
        :meth:`KdTreeFLANN.nearest_k_search_for_cloud`."""
        return _k_search_cloud(self._ready(), pc, k)

    def nearest_k_search_for_point(self, PointCloud pc not None,
                                   Py_ssize_t index, int k=1):
        """k nearest neighbours of a single point of *pc*."""
        return _k_search_point(self._ready(), pc, index, k)

    def radius_search_for_cloud(self, PointCloud pc not None, double radius,
                                unsigned int max_nn=0):
        """Neighbours within *radius* for every point of *pc*, padded to
        ``(len(pc), max_nn)``."""
        return _radius_search_cloud(self._ready(), pc, radius, max_nn)
