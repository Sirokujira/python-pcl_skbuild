# distutils: language = c++
# cython: language_level=3
"""Octree wrappers, named after sirokujira/python-pcl (pcl/pxi/Octree/).

Reach them through `PointCloud.make_octreeSearch(resolution)` and
`PointCloud.make_octreeChangeDetector(resolution)`.

An octree is built in two steps — `set_InputCloud` then
`add_points_from_input_cloud` — because PCL keeps the tree and the cloud
separate; forgetting the second step gives an empty tree rather than an
error, so the wrappers do it for you when constructed from a cloud.

`get_occupied_voxel_centers` goes through pcl/compat/eigen_results.h:
PCL fills a `std::vector<PointT, Eigen::aligned_allocator<PointT>>`,
which is a different C++ type from the `std::vector` Cython can name.
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.octree.octree_search cimport (
    OctreePointCloudSearch as cOctreePointCloudSearch)
from pcl.pxd.octree.octree_pointcloud_changedetector cimport (
    OctreePointCloudChangeDetector as cOctreePointCloudChangeDetector)
from pcl.pxd.compat.eigen_results cimport occupiedVoxelCenters

from pcl._pointcloud cimport PointCloud


cdef class OctreePointCloudSearch:
    """Voxel-tree search over a cloud (pcl::octree::OctreePointCloudSearch)."""

    cdef cOctreePointCloudSearch[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudSearch[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        cdef cOctreePointCloudSearch[PointXYZ]* octree = self.me
        with nogil:
            octree.addPointsFromInputCloud()

    def define_bounding_box(self, *bounds):
        """No arguments: fit the box to the input cloud. Six arguments:
        ``min_x, min_y, min_z, max_x, max_y, max_z``."""
        if not bounds:
            self.me.defineBoundingBox()
        elif len(bounds) == 6:
            self.me.defineBoundingBox(bounds[0], bounds[1], bounds[2],
                                      bounds[3], bounds[4], bounds[5])
        else:
            raise TypeError(
                "define_bounding_box takes 0 or 6 arguments, got %d"
                % len(bounds))

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def voxel_search(self, PointCloud pc not None, Py_ssize_t index):
        """Indices of every point sharing a voxel with ``pc[index]``."""
        cdef cPointCloud[PointXYZ]* cloud = pc.ptr()
        cdef Py_ssize_t n = <Py_ssize_t> cloud.size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef vector[int] indices
        self.me.voxelSearch(deref(cloud)[<size_t> index], indices)
        return [indices[i] for i in range(indices.size())]

    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """k nearest neighbours for every point of *pc*.

        Returns ``(indices, sqr_distances)`` of shape ``(len(pc), k)``.
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

    def radius_search(self, PointCloud pc not None, Py_ssize_t index,
                      double radius, unsigned int max_nn=0):
        """Neighbours of ``pc[index]`` within *radius*.

        Returns ``(indices, sqr_distances)`` as plain lists — unlike the
        whole-cloud searches there is no rectangular shape to fill, so
        the true neighbour count comes back as-is.
        """
        cdef cPointCloud[PointXYZ]* cloud = pc.ptr()
        cdef Py_ssize_t n = <Py_ssize_t> cloud.size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")

        cdef vector[int] indices
        cdef vector[float] sqr_distances
        cdef int found = self.me.radiusSearch(
            deref(cloud)[<size_t> index], radius, indices, sqr_distances,
            max_nn)
        return ([indices[i] for i in range(found)],
                [sqr_distances[i] for i in range(found)])

    def get_occupied_voxel_centers(self):
        """Centre point of every occupied voxel, as a list of triples."""
        cdef vector[PointXYZ] centers
        occupiedVoxelCenters(deref(self.me), centers)
        return [(centers[i].x, centers[i].y, centers[i].z)
                for i in range(centers.size())]


cdef class OctreePointCloudChangeDetector:
    """Spatial change detection between two clouds
    (pcl::octree::OctreePointCloudChangeDetector).

    Fill from the first cloud, `switch_buffers()`, fill from the second,
    then `get_PointIndicesFromNewVoxels()` gives the points of the second
    cloud that fell in voxels the first did not occupy.
    """

    cdef cOctreePointCloudChangeDetector[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudChangeDetector[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        cdef cOctreePointCloudChangeDetector[PointXYZ]* octree = self.me
        with nogil:
            octree.addPointsFromInputCloud()

    def switch_buffers(self):
        """Keep the current tree as the reference and start a fresh one."""
        self.me.switchBuffers()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_PointIndicesFromNewVoxels(self, int min_points_per_leaf=0):
        cdef vector[int] indices
        self.me.getPointIndicesFromNewVoxels(indices, min_points_per_leaf)
        return [indices[i] for i in range(indices.size())]
