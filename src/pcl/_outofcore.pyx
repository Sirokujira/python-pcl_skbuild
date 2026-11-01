# distutils: language = c++
# cython: language_level=3
"""pcl/outofcore: a disk-backed octree for clouds larger than memory.

Built only where PCL has its outofcore module, which PCL builds only
with VTK (its templates include VTK headers); `pcl.OutofcoreOctree` is
absent otherwise. Goes through pcl/compat/outofcore_args.h, which also
explains why every change closes and reopens the tree.
"""

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.compat.outofcore_args cimport OutofcoreTree

from pcl._pointcloud cimport PointCloud


cdef _box(values, str what):
    import numpy as np
    arr = np.ascontiguousarray(values, dtype=np.float64).ravel()
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError("%s must be finite (x, y, z)" % what)
    return arr


cdef class OutofcoreOctree:
    """pcl::outofcore::OutofcoreOctreeBase on PointXYZ.

        tree = pcl.OutofcoreOctree.create("tree/tree.oct_idx",
                                          (0, 0, 0), (100, 100, 100), depth=4)
        tree.add(cloud)                  # stored in the leaves, on disk
        near = tree.query((0, 0, 0), (10, 10, 10))
        tree = pcl.OutofcoreOctree.open("tree/tree.oct_idx")

    The index file's directory holds one subdirectory per node; it must
    be new for `create`. Points outside the box are not stored. Queries
    read the leaves (`depth` = the tree's) or, after `build_LOD`, a
    subsample at any shallower depth.
    """

    cdef OutofcoreTree* me

    def __cinit__(self):
        self.me = new OutofcoreTree()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    @staticmethod
    def create(path, minimum, maximum, unsigned int depth):
        import os
        path = os.fspath(path)
        if not str(path).endswith(".oct_idx"):
            raise ValueError("the index file must end in .oct_idx")
        if os.path.exists(path):
            raise ValueError("%s already exists; use open()" % path)
        lo = _box(minimum, "minimum")
        hi = _box(maximum, "maximum")
        if not (hi > lo).all():
            raise ValueError("maximum must exceed minimum on every axis")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        cdef OutofcoreOctree tree = OutofcoreOctree()
        cdef double[::1] vlo = lo
        cdef double[::1] vhi = hi
        tree.me.create(os.fsencode(os.path.abspath(path)), &vlo[0], &vhi[0],
                       depth)
        return tree

    @staticmethod
    def open(path):
        import os
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise IOError("no octree index at %s" % path)
        cdef OutofcoreOctree tree = OutofcoreOctree()
        tree.me.open(os.fsencode(os.path.abspath(path)))
        return tree

    def add(self, PointCloud cloud not None):
        """Store the points; returns how many were inside the box."""
        if cloud.size == 0:
            return 0
        return self.me.add(cloud.thisptr_shared)

    def build_LOD(self):
        """Fill each shallower level with a subsample of the one below
        (PCL's level of detail), so `query` at depth d returns a coarse
        version of the cloud."""
        self.me.buildLOD()

    def query(self, minimum, maximum, depth=None):
        """The points inside the box, from `depth` (default the leaves)."""
        lo = _box(minimum, "minimum")
        hi = _box(maximum, "maximum")
        cdef unsigned int d = self.me.depth() if depth is None else depth
        if d > self.me.depth():
            raise ValueError("depth must be <= %d" % self.me.depth())
        cdef double[::1] vlo = lo
        cdef double[::1] vhi = hi
        cdef PointCloud out = PointCloud()
        self.me.query(&vlo[0], &vhi[0], d, deref(out.ptr()))
        return out

    def get_OccupiedVoxelCenters(self, depth=None):
        import numpy as np
        cdef unsigned int d = self.me.depth() if depth is None else depth
        if d > self.me.depth():
            raise ValueError("depth must be <= %d" % self.me.depth())
        cdef vector[double] out
        self.me.voxelCentres(d, out)
        return np.asarray(out, dtype=np.float64).reshape(-1, 3)

    @property
    def depth(self):
        return self.me.depth()

    def get_VoxelSideLength(self, depth=None):
        cdef unsigned int d = self.me.depth() if depth is None else depth
        return self.me.voxelSideLength(d)

    def get_NumPointsAtDepth(self, unsigned int depth):
        return self.me.pointsAtDepth(depth)

    def get_BoundingBox(self):
        import numpy as np
        lo = np.empty(3)
        hi = np.empty(3)
        cdef double[::1] vlo = lo
        cdef double[::1] vhi = hi
        self.me.boundingBox(&vlo[0], &vhi[0])
        return lo, hi
