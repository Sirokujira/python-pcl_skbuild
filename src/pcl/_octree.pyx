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
from pcl.pxd.octree.octree_pointcloud_density cimport (
    OctreePointCloudDensity as cOctreePointCloudDensity)
from pcl.pxd.compat.octree_args cimport (
    VoxelCentroidOctree, intersectedVoxelCenters, intersectedVoxelIndices,
    occupiedVoxelCentersOf, voxelCentroids)

from pcl._pointcloud cimport PointCloud
from libcpp.string cimport string
from pcl.pxd.point_types cimport PointXYZRGB
from pcl.pxd.compression.octree_pointcloud_compression cimport (
    OctreePointCloudCompression as cOctreePointCloudCompression)
cimport pcl.pxd.compression.compression_profiles as profiles
from pcl.pxd.compat.compression_args cimport compressCloud, decompressCloud
from pcl._pointtypes cimport PointCloud_PointXYZRGB, PointCloud_PointXYZRGBA
from pcl._pointtypes cimport wrap_xyzrgba_cloud
from libcpp.memory cimport shared_ptr
from pcl.pxd.point_types cimport PointXYZRGBA
from pcl.pxd.octree.octree_pointcloud_occupancy cimport (
    OctreePointCloudOccupancy as cOctreePointCloudOccupancy)
from pcl.pxd.octree.octree_pointcloud_pointvector cimport (
    OctreePointCloudPointVector as cOctreePointCloudPointVector)
from pcl.pxd.octree.octree_pointcloud_singlepoint cimport (
    OctreePointCloudSinglePoint as cOctreePointCloudSinglePoint)
from pcl.pxd.octree.octree_pointcloud_adjacency cimport (
    OctreePointCloudAdjacency as cOctreePointCloudAdjacency)
from pcl.pxd.compat.octree_args cimport (
    adjacencyGraph, adjacencyOccluded, leafCenters, leafIndices)
from pcl.pxd.compat.compression_args cimport (
    compressOrganized, decompressOrganized)


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

    def is_voxel_occupied_at_point(self, point):
        """True when the voxel containing ``(x, y, z)`` holds any point."""
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def delete_voxel_at_point(self, point):
        """Drop the voxel containing ``(x, y, z)``, and the points it
        indexed, from the tree. The input cloud itself is untouched."""
        cdef float x, y, z
        x, y, z = point
        cdef PointXYZ p
        p.x = x
        p.y = y
        p.z = z
        self.me.deleteVoxelAtPoint(p)

    def get_intersected_voxel_centers(self, origin, direction,
                                      unsigned int max_voxel_count=0):
        """Centres of the occupied voxels a ray from *origin* along
        *direction* passes through, nearest first, as an ``(n, 3)``
        float32 array. `max_voxel_count` 0 means all of them."""
        cdef float o[3]
        cdef float d[3]
        _ray(origin, direction, o, d)
        cdef vector[PointXYZ] centers
        with nogil:
            intersectedVoxelCenters(deref(self.me), o, d, max_voxel_count,
                                    centers)
        return _points_array(centers)

    def get_intersected_voxel_indices(self, origin, direction,
                                      unsigned int max_voxel_count=0):
        """Indices of the cloud's points inside the voxels the ray passes
        through, nearest voxel first, as an int32 array."""
        import numpy as np
        cdef float o[3]
        cdef float d[3]
        _ray(origin, direction, o, d)
        cdef vector[int] indices
        with nogil:
            intersectedVoxelIndices(deref(self.me), o, d, max_voxel_count,
                                    indices)
        result = np.empty(indices.size(), dtype=np.int32)
        cdef int[::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> indices.size()):
            view[i] = indices[i]
        return result


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


# Compression profiles, re-exported from the header so the values
# cannot drift.
LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR = profiles.LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
LOW_RES_ONLINE_COMPRESSION_WITH_COLOR = profiles.LOW_RES_ONLINE_COMPRESSION_WITH_COLOR
MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR = profiles.MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
MED_RES_ONLINE_COMPRESSION_WITH_COLOR = profiles.MED_RES_ONLINE_COMPRESSION_WITH_COLOR
HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR = profiles.HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR = profiles.HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR
LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR = profiles.LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
LOW_RES_OFFLINE_COMPRESSION_WITH_COLOR = profiles.LOW_RES_OFFLINE_COMPRESSION_WITH_COLOR
MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR = profiles.MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
MED_RES_OFFLINE_COMPRESSION_WITH_COLOR = profiles.MED_RES_OFFLINE_COMPRESSION_WITH_COLOR
HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR = profiles.HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
HIGH_RES_OFFLINE_COMPRESSION_WITH_COLOR = profiles.HIGH_RES_OFFLINE_COMPRESSION_WITH_COLOR
MANUAL_CONFIGURATION = profiles.MANUAL_CONFIGURATION

cdef class OctreePointCloudCompression:
    """Octree point cloud compression (pcl::io::OctreePointCloudCompression)
    for :class:`PointCloud`.

    `encode()` turns a cloud into bytes and `decode()` turns them back.
    The codec is STATEFUL -- frames after the first are coded as
    differences every `iframe_rate` frames -- so use one instance to
    encode a stream and a separate one to decode it, in order. Positions
    come back quantised to `point_resolution`.

    Profiles (`pcl.MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR`, ...) fix the
    resolutions; `pcl.MANUAL_CONFIGURATION` uses the arguments instead.
    """

    cdef cOctreePointCloudCompression[PointXYZ]* me

    def __cinit__(self, int profile=-1, bint show_statistics=False,
                  double point_resolution=0.001,
                  double octree_resolution=0.01,
                  bint voxel_grid_downsampling=False,
                  unsigned int iframe_rate=30, bint color_encoding=True,
                  int color_bit_resolution=6):
        if profile == -1:
            profile = profiles.MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
        # COMPRESSION_PROFILE_COUNT sits between the profiles and
        # MANUAL_CONFIGURATION; PCL indexes its profile table with any
        # other value unchecked.
        if (not profiles.LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR <= profile
                <= profiles.MANUAL_CONFIGURATION
                or profile == profiles.COMPRESSION_PROFILE_COUNT):
            raise ValueError("unknown compression profile %d" % profile)
        if point_resolution <= 0 or octree_resolution <= 0:
            raise ValueError("resolutions must be positive")
        if not 1 <= color_bit_resolution <= 8:
            raise ValueError("color_bit_resolution must be 1..8")
        self.me = new cOctreePointCloudCompression[PointXYZ](
            <profiles.compression_Profiles_e> profile, show_statistics,
            point_resolution, octree_resolution, voxel_grid_downsampling,
            iframe_rate, color_encoding,
            <unsigned char> color_bit_resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def encode(self, PointCloud cloud not None):
        """Return the cloud compressed, as bytes."""
        if cloud.size == 0:
            raise ValueError("cannot encode an empty cloud")
        cdef string data
        data = compressCloud[PointXYZ](deref(self.me), cloud.thisptr_shared)
        return <bytes> data

    def decode(self, data):
        """Return the cloud `encode()` produced from *data* (bytes)."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("expected bytes, got %s" % type(data).__name__)
        cdef string s = bytes(data)
        if s.empty():
            raise ValueError("cannot decode an empty buffer")
        cdef PointCloud cloud = PointCloud()
        cdef cPointCloud[PointXYZ]* out = cloud.ptr()
        decompressCloud[PointXYZ](deref(self.me), s, deref(out))
        return cloud


cdef class OctreePointCloudCompressionRGB:
    """Octree point cloud compression (pcl::io::OctreePointCloudCompression)
    for :class:`PointCloud_PointXYZRGB`, colour included
    (`color_bit_resolution` bits per channel).

    `encode()` turns a cloud into bytes and `decode()` turns them back.
    The codec is STATEFUL -- frames after the first are coded as
    differences every `iframe_rate` frames -- so use one instance to
    encode a stream and a separate one to decode it, in order. Positions
    come back quantised to `point_resolution`.

    Profiles (`pcl.MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR`, ...) fix the
    resolutions; `pcl.MANUAL_CONFIGURATION` uses the arguments instead.
    """

    cdef cOctreePointCloudCompression[PointXYZRGB]* me

    def __cinit__(self, int profile=-1, bint show_statistics=False,
                  double point_resolution=0.001,
                  double octree_resolution=0.01,
                  bint voxel_grid_downsampling=False,
                  unsigned int iframe_rate=30, bint color_encoding=True,
                  int color_bit_resolution=6):
        if profile == -1:
            profile = profiles.MED_RES_ONLINE_COMPRESSION_WITH_COLOR
        # COMPRESSION_PROFILE_COUNT sits between the profiles and
        # MANUAL_CONFIGURATION; PCL indexes its profile table with any
        # other value unchecked.
        if (not profiles.LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR <= profile
                <= profiles.MANUAL_CONFIGURATION
                or profile == profiles.COMPRESSION_PROFILE_COUNT):
            raise ValueError("unknown compression profile %d" % profile)
        if point_resolution <= 0 or octree_resolution <= 0:
            raise ValueError("resolutions must be positive")
        if not 1 <= color_bit_resolution <= 8:
            raise ValueError("color_bit_resolution must be 1..8")
        self.me = new cOctreePointCloudCompression[PointXYZRGB](
            <profiles.compression_Profiles_e> profile, show_statistics,
            point_resolution, octree_resolution, voxel_grid_downsampling,
            iframe_rate, color_encoding,
            <unsigned char> color_bit_resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def encode(self, PointCloud_PointXYZRGB cloud not None):
        """Return the cloud compressed, as bytes."""
        if cloud.size == 0:
            raise ValueError("cannot encode an empty cloud")
        cdef string data
        data = compressCloud[PointXYZRGB](deref(self.me), cloud.thisptr_shared)
        return <bytes> data

    def decode(self, data):
        """Return the cloud `encode()` produced from *data* (bytes)."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("expected bytes, got %s" % type(data).__name__)
        cdef string s = bytes(data)
        if s.empty():
            raise ValueError("cannot decode an empty buffer")
        cdef PointCloud_PointXYZRGB cloud = PointCloud_PointXYZRGB()
        cdef cPointCloud[PointXYZRGB]* out = cloud.ptr()
        decompressCloud[PointXYZRGB](deref(self.me), s, deref(out))
        return cloud


cdef _ray(origin, direction, float* o, float* d):
    ox, oy, oz = origin
    dx, dy, dz = direction
    if dx == 0 and dy == 0 and dz == 0:
        raise ValueError("direction must not be the zero vector")
    o[0] = ox
    o[1] = oy
    o[2] = oz
    d[0] = dx
    d[1] = dy
    d[2] = dz


cdef object _points_array(vector[PointXYZ]& points):
    import numpy as np
    result = np.empty((points.size(), 3), dtype=np.float32)
    cdef float[:, ::1] view = result
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t> points.size()):
        view[i, 0] = points[i].x
        view[i, 1] = points[i].y
        view[i, 2] = points[i].z
    return result


cdef PointXYZ _point(point) except *:
    cdef PointXYZ p
    x, y, z = point
    p.x = x
    p.y = y
    p.z = z
    return p


cdef class OctreePointCloudVoxelCentroid:
    """An octree whose leaves keep the centroid of their points
    (pcl::octree::OctreePointCloudVoxelCentroid) — a voxel-grid filter
    you can also query by position.

        octree = cloud.make_octreeVoxelCentroid(0.05)
        centroids = octree.get_voxel_centroids()

    Where the voxel boundaries fall: unless `define_bounding_box` was
    called before adding points, PCL aligns the grid on the FIRST point
    added, which lands exactly on a voxel corner — so points clustered
    around it are split across up to eight voxels. Construct without a
    cloud and call `define_bounding_box` first to choose the grid.
    """

    cdef VoxelCentroidOctree* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new VoxelCentroidOctree(resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def define_bounding_box(self, min_point, max_point):
        """Fix the octree's extent (and with it the voxel grid) before any
        points are added. PCL grows it to a power-of-two number of voxels
        from `min_point`."""
        cdef double x0, y0, z0, x1, y1, z1
        x0, y0, z0 = min_point
        x1, y1, z1 = max_point
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError("max_point must exceed min_point on every axis")
        if self.me.getLeafCount() > 0:
            raise RuntimeError(
                "define_bounding_box must come before any points are added")
        self.me.defineBoundingBox(x0, y0, z0, x1, y1, z1)

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def get_voxel_centroids(self):
        """One centroid per occupied voxel, as an ``(n, 3)`` float32
        array."""
        cdef vector[PointXYZ] centroids
        with nogil:
            voxelCentroids(deref(self.me), centroids)
        return _points_array(centroids)

    def get_voxel_centroid_at_point(self, point):
        """Centroid of the voxel containing ``(x, y, z)`` as a tuple, or
        None when that voxel is empty."""
        cdef PointXYZ query = _point(point)
        cdef PointXYZ centroid
        if not self.me.centroidAtPoint(query, centroid):
            return None
        return (centroid.x, centroid.y, centroid.z)

    def get_occupied_voxel_centers(self):
        """Geometric centre of every occupied voxel, ``(n, 3)``."""
        cdef vector[PointXYZ] centers
        occupiedVoxelCentersOf(deref(self.me), centers)
        return _points_array(centers)


cdef class OctreePointCloudDensity:
    """An octree whose leaves count their points
    (pcl::octree::OctreePointCloudDensity).

        octree = cloud.make_octreeDensity(0.05)
        octree.get_voxel_density_at_point((0.1, 0.2, 0.0))

    The grid is aligned on the first point added unless
    `define_bounding_box` came first; see
    :class:`OctreePointCloudVoxelCentroid`.
    """

    cdef cOctreePointCloudDensity[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudDensity[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def define_bounding_box(self, min_point, max_point):
        """Fix the octree's extent (and with it the voxel grid) before any
        points are added. PCL grows it to a power-of-two number of voxels
        from `min_point`."""
        cdef double x0, y0, z0, x1, y1, z1
        x0, y0, z0 = min_point
        x1, y1, z1 = max_point
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError("max_point must exceed min_point on every axis")
        if self.me.getLeafCount() > 0:
            raise RuntimeError(
                "define_bounding_box must come before any points are added")
        self.me.defineBoundingBox(x0, y0, z0, x1, y1, z1)

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def get_voxel_density_at_point(self, point):
        """Number of points in the voxel containing ``(x, y, z)`` (0 when
        it is empty)."""
        cdef PointXYZ query = _point(point)
        return self.me.getVoxelDensityAtPoint(query)

    def get_occupied_voxel_centers(self):
        """Geometric centre of every occupied voxel, ``(n, 3)``."""
        cdef vector[PointXYZ] centers
        occupiedVoxelCentersOf(deref(self.me), centers)
        return _points_array(centers)


ctypedef fused _VariantOctree:
    cOctreePointCloudOccupancy[PointXYZ]
    cOctreePointCloudPointVector[PointXYZ]
    cOctreePointCloudSinglePoint[PointXYZ]


cdef _define_bounding_box(_VariantOctree* octree, min_point, max_point):
    cdef double x0, y0, z0, x1, y1, z1
    x0, y0, z0 = min_point
    x1, y1, z1 = max_point
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        raise ValueError("max_point must exceed min_point on every axis")
    if octree.getLeafCount() > 0:
        raise RuntimeError(
            "define_bounding_box must come before any points are added")
    octree.defineBoundingBox(x0, y0, z0, x1, y1, z1)


cdef object _float_rows(vector[float]& values, int width):
    import numpy as np
    out = np.empty(values.size(), dtype=np.float32)
    cdef float[::1] view = out
    cdef size_t i
    for i in range(values.size()):
        view[i] = values[i]
    return out.reshape(-1, width)


cdef list _split(vector[int]& flat, vector[int]& counts):
    import numpy as np
    cdef list out = []
    cdef size_t i, j, start = 0
    for i in range(counts.size()):
        one = np.empty(counts[i], dtype=np.int32)
        for j in range(<size_t> counts[i]):
            one[j] = flat[start + j]
        start += counts[i]
        out.append(one)
    return out


cdef class OctreePointCloudOccupancy:
    """An octree that only records which voxels are occupied
    (pcl::octree::OctreePointCloudOccupancy): voxels can be marked
    without storing any point, the cheapest occupancy grid PCL has."""

    cdef cOctreePointCloudOccupancy[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudOccupancy[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def define_bounding_box(self, min_point, max_point):
        """Fix the octree's extent before any points are added; see
        :class:`OctreePointCloudVoxelCentroid`."""
        _define_bounding_box(self.me, min_point, max_point)

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def delete_voxel_at_point(self, point):
        self.me.deleteVoxelAtPoint(_point(point))

    def set_occupied_voxel_at_point(self, point):
        """Mark the voxel holding ``(x, y, z)`` occupied, storing nothing."""
        self.me.setOccupiedVoxelAtPoint(_point(point))

    def set_occupied_voxels_at_points_from_cloud(self, PointCloud pc not None):
        self.me.setOccupiedVoxelsAtPointsFromCloud(pc.thisptr_shared)

    def get_occupied_voxel_centers(self):
        """Centre of every occupied voxel, ``(n, 3)`` float32."""
        cdef vector[float] centers
        leafCenters(deref(self.me), centers)
        return _float_rows(centers, 3)


cdef class OctreePointCloudPointVector:
    """An octree whose leaves keep the indices of every point in
    them (pcl::octree::OctreePointCloudPointVector): `get_voxels()`
    returns each voxel's points."""

    cdef cOctreePointCloudPointVector[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudPointVector[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def define_bounding_box(self, min_point, max_point):
        """Fix the octree's extent before any points are added; see
        :class:`OctreePointCloudVoxelCentroid`."""
        _define_bounding_box(self.me, min_point, max_point)

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def delete_voxel_at_point(self, point):
        self.me.deleteVoxelAtPoint(_point(point))

    def get_voxels(self):
        """``(centers, indices)``: the ``(n, 3)`` centre of every occupied
        voxel and, per voxel, the indices of all its points."""
        cdef vector[int] flat
        cdef vector[int] counts
        cdef vector[float] centers
        leafIndices(deref(self.me), flat, counts, centers)
        return _float_rows(centers, 3), _split(flat, counts)

    def get_occupied_voxel_centers(self):
        return self.get_voxels()[0]


cdef class OctreePointCloudSinglePoint:
    """An octree whose leaves keep one point index -- the last added
    (pcl::octree::OctreePointCloudSinglePoint): a one-point-per-voxel
    downsampling that keeps original points rather than centroids."""

    cdef cOctreePointCloudSinglePoint[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudSinglePoint[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def define_bounding_box(self, min_point, max_point):
        """Fix the octree's extent before any points are added; see
        :class:`OctreePointCloudVoxelCentroid`."""
        _define_bounding_box(self.me, min_point, max_point)

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def delete_voxel_at_point(self, point):
        self.me.deleteVoxelAtPoint(_point(point))

    def get_voxels(self):
        """``(centers, indices)``: the ``(n, 3)`` centre of every occupied
        voxel and the index of the one point it keeps (the last one
        added), ``(n,)`` int32."""
        import numpy as np
        cdef vector[int] flat
        cdef vector[int] counts
        cdef vector[float] centers
        leafIndices(deref(self.me), flat, counts, centers)
        return (_float_rows(centers, 3),
                np.asarray([flat[i] for i in range(flat.size())], np.int32))

    def get_occupied_voxel_centers(self):
        return self.get_voxels()[0]


cdef class OctreePointCloudAdjacency:
    """An octree whose leaves know their 26-neighbourhood
    (pcl::octree::OctreePointCloudAdjacency): `get_adjacency()` returns
    the voxel graph. Its grid always starts at the cloud's minimum x, y,
    z -- PCL resets the bounding box from the points when they are added,
    so there is no `define_bounding_box` here."""

    cdef cOctreePointCloudAdjacency[PointXYZ]* me

    def __cinit__(self, double resolution, PointCloud pc=None):
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        self.me = new cOctreePointCloudAdjacency[PointXYZ](resolution)
        if pc is not None:
            self.set_InputCloud(pc)
            self.add_points_from_input_cloud()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def add_points_from_input_cloud(self):
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        self.me.deleteTree()

    def get_resolution(self):
        return self.me.getResolution()

    def get_tree_depth(self):
        return self.me.getTreeDepth()

    def get_leaf_count(self):
        return self.me.getLeafCount()

    def is_voxel_occupied_at_point(self, point):
        cdef double x, y, z
        x, y, z = point
        return self.me.isVoxelOccupiedAtPoint(x, y, z)

    def delete_voxel_at_point(self, point):
        self.me.deleteVoxelAtPoint(_point(point))

    def get_adjacency(self):
        """``(centers, counts, edges)``: per occupied voxel its ``(n, 3)``
        centre and point count, and every pair of neighbouring voxels
        (26-connectivity) once, ``(m, 2)`` int32 indices into the first
        two -- the graph SupervoxelClustering grows over."""
        import numpy as np
        cdef vector[float] centers
        cdef vector[int] counts
        cdef vector[int] edges
        adjacencyGraph(deref(self.me), centers, counts, edges)
        return (_float_rows(centers, 3),
                np.asarray([counts[i] for i in range(counts.size())], np.int32),
                np.asarray([edges[i] for i in range(edges.size())],
                           np.int32).reshape(-1, 2))

    def get_occupied_voxel_centers(self):
        return self.get_adjacency()[0]

    def test_for_occlusion(self, point, camera=(0.0, 0.0, 0.0)):
        """Whether an occupied voxel lies between `camera` and ``(x, y,
        z)`` (PCL's testForOcclusion walk). The walk stops at the octree's
        bounding box: PCL's own keeps going, wraps its voxel keys outside
        the box and reports occlusions by nothing whenever the camera is
        outside it -- the usual case."""
        return adjacencyOccluded(deref(self.me), _point(point),
                                 _point(camera))


cdef class OrganizedPointCloudCompression:
    """Lossless-in-structure compression of ORGANIZED clouds
    (pcl::io::OrganizedPointCloudCompression): the depth image, and the
    colour image when asked, PNG-encoded. Unlike
    :class:`OctreePointCloudCompression` it keeps every pixel and needs
    no state across frames. Depth is stored as a Kinect-style disparity
    image quantised for the cloud's focal length, which the encoder
    estimates: about 2 mm at 1 m for f = 525 px, coarser for a narrower
    focal length; x and y are rebuilt from the pixel grid. PCL's decoder
    returns the cloud unorganized (it fills it with push_back, which
    resets the height); the wrapper restores the size from the stream.

        data = codec.encode(cloud)            # bytes
        back = codec.decode(data, color=True) # PointCloud_PointXYZRGBA
    """

    def encode(self, cloud, color=None, bint mono=False, int png_level=-1):
        """Compress a :class:`PointCloud` or :class:`PointCloud_PointXYZRGBA`;
        `color` (default: whether the cloud has colour) adds the colour
        image, `mono` stores it as grey, `png_level` is libpng's 0..9
        (-1 its default)."""
        if cloud.height <= 1:
            raise ValueError("the cloud must be organized (height > 1)")
        if not -1 <= png_level <= 9:
            raise ValueError("png_level must be -1 or 0..9")
        cdef string data
        cdef PointCloud xyz
        cdef PointCloud_PointXYZRGBA rgba
        if isinstance(cloud, PointCloud_PointXYZRGBA):
            rgba = cloud
            data = compressOrganized(rgba.thisptr_shared,
                                     True if color is None else bool(color),
                                     mono, png_level)
        elif isinstance(cloud, PointCloud):
            if color:
                raise ValueError("a PointCloud has no colour to encode")
            xyz = cloud
            data = compressOrganized(xyz.thisptr_shared, False, False,
                                     png_level)
        else:
            raise TypeError("expected a PointCloud or PointCloud_PointXYZRGBA")
        return <bytes> data

    def decode(self, data, bint color=False):
        """The organized cloud `encode()` produced: a
        :class:`PointCloud_PointXYZRGBA` with `color`, else a
        :class:`PointCloud`."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("expected bytes, got %s" % type(data).__name__)
        cdef string s = bytes(data)
        cdef PointCloud xyz
        cdef shared_ptr[cPointCloud[PointXYZRGBA]] holder
        cdef bint ok
        if color:
            holder.reset(new cPointCloud[PointXYZRGBA]())
            ok = decompressOrganized(s, deref(holder.get()))
            if not ok:
                raise ValueError("not an organized point cloud stream")
            return wrap_xyzrgba_cloud(holder)
        xyz = PointCloud()
        ok = decompressOrganized(s, deref(xyz.ptr()))
        if not ok:
            raise ValueError("not an organized point cloud stream")
        return xyz

