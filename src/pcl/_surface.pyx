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

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport Normal, PointXYZ, PointNormal
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.surface.mls cimport MovingLeastSquares as cMovingLeastSquares
from pcl.pxd.surface.concave_hull cimport ConcaveHull as cConcaveHull
from pcl.pxd.surface.convex_hull cimport ConvexHull as cConvexHull
from pcl.pxd.surface.gp3 cimport (
    GreedyProjectionTriangulation as cGreedyProjectionTriangulation)
from pcl.pxd.vertices cimport Vertices
from pcl.pxd.compat.mesh_args cimport (
    MESH_FORMAT_OBJ, MESH_FORMAT_PLY, MESH_FORMAT_VTK, loadMesh,
    saveMesh)

from pcl._pointcloud cimport PointCloud, wrap_cloud
from pcl._pointtypes cimport PointCloud_Normal


cdef object _polygon_list(vector[Vertices]& polygons):
    """PCL's vector<Vertices> as a list of index tuples.

    Plain loops on purpose: a comprehension or generator over a C++
    reference makes Cython build a closure holding that reference, and
    the generated scope object segfaults on construction.
    """
    cdef size_t i, j
    result = []
    for i in range(polygons.size()):
        polygon = []
        for j in range(polygons[i].vertices.size()):
            polygon.append(polygons[i].vertices[j])
        result.append(tuple(polygon))
    return result


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

    def reconstruct_with_polygons(self):
        """Return ``(points, polygons)``; each polygon is a tuple of
        indices into *points*. That pair is what `CropHull` takes."""
        cdef PointCloud result = PointCloud()
        cdef cPointCloud[PointXYZ]* out = result.ptr()
        cdef cConcaveHull[PointXYZ]* hull = self.me
        cdef vector[Vertices] polygons
        with nogil:
            hull.reconstruct(deref(out), polygons)
        return result, _polygon_list(polygons)


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

    def reconstruct_with_polygons(self):
        """Return ``(points, polygons)``; each polygon is a tuple of
        indices into *points*. That pair is what `CropHull` takes."""
        cdef PointCloud result = PointCloud()
        cdef cPointCloud[PointXYZ]* out = result.ptr()
        cdef cConvexHull[PointXYZ]* hull = self.me
        cdef vector[Vertices] polygons
        with nogil:
            hull.reconstruct(deref(out), polygons)
        return result, _polygon_list(polygons)


cdef class GreedyProjectionTriangulation:
    """Triangulate a cloud that has normals (pcl::GreedyProjectionTriangulation).

    Meshing needs both geometry and orientation, and PCL wants them in
    one PointNormal cloud; `set_InputCloud` takes the two separately and
    joins them, so callers never build that type by hand:

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        gp3 = cloud.make_GreedyProjectionTriangulation()
        gp3.set_InputCloud(cloud, normals.compute_cloud())
        gp3.set_SearchRadius(0.5)
        triangles = gp3.reconstruct()

    `reconstruct` returns index triples into the ORIGINAL cloud, so the
    points never leave Python's side of the boundary.
    """

    cdef cGreedyProjectionTriangulation[PointNormal]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cGreedyProjectionTriangulation[PointNormal]()
        self.has_input = False
        # A cloud alone is not enough input, so a constructor argument
        # only records intent; set_InputCloud does the real work.

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None,
                       PointCloud_Normal normals not None):
        """Join *pc* and *normals* into the PointNormal cloud PCL wants."""
        cdef Py_ssize_t n = <Py_ssize_t> pc.ptr().size()
        if <Py_ssize_t> normals.ptr().size() != n:
            raise ValueError(
                "cloud and normals must have the same length, got %d and %d"
                % (n, normals.ptr().size()))

        cdef shared_ptr[cPointCloud[PointNormal]] joined
        joined.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = joined.get()
        out.resize(<size_t> n)
        out.width = <unsigned int> n
        out.height = 1

        cdef cPointCloud[PointXYZ]* src = pc.ptr()
        cdef cPointCloud[Normal]* nrm = normals.ptr()
        cdef PointNormal* dst
        cdef Py_ssize_t i
        for i in range(n):
            dst = &(deref(out)[<size_t> i])
            dst.x = deref(src)[<size_t> i].x
            dst.y = deref(src)[<size_t> i].y
            dst.z = deref(src)[<size_t> i].z
            dst.normal_x = deref(nrm)[<size_t> i].normal_x
            dst.normal_y = deref(nrm)[<size_t> i].normal_y
            dst.normal_z = deref(nrm)[<size_t> i].normal_z
            dst.curvature = deref(nrm)[<size_t> i].curvature

        self.me.setInputCloud(joined)
        self.has_input = True

    def set_SearchRadius(self, double radius):
        """Longest edge a triangle may have. Required."""
        self.me.setSearchRadius(radius)

    def get_SearchRadius(self):
        return self.me.getSearchRadius()

    def set_Mu(self, double mu):
        """Neighbour distance multiplier, relative to the nearest one."""
        self.me.setMu(mu)

    def get_Mu(self):
        return self.me.getMu()

    def set_MaximumNearestNeighbors(self, int count):
        self.me.setMaximumNearestNeighbors(count)

    def get_MaximumNearestNeighbors(self):
        return self.me.getMaximumNearestNeighbors()

    def set_MaximumSurfaceAngle(self, double angle):
        self.me.setMaximumSurfaceAngle(angle)

    def set_MinimumAngle(self, double angle):
        self.me.setMinimumAngle(angle)

    def set_MaximumAngle(self, double angle):
        self.me.setMaximumAngle(angle)

    def set_NormalConsistency(self, bint consistent):
        self.me.setNormalConsistency(consistent)

    def reconstruct(self):
        """Return the triangles as a list of index triples."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(cloud, normals) is required before "
                "reconstruct()")
        cdef vector[Vertices] polygons
        cdef cGreedyProjectionTriangulation[PointNormal]* mesher = self.me
        with nogil:
            mesher.reconstruct(polygons)
        return _polygon_list(polygons)


# Mesh file formats, by extension. VTK is write-only: PCL's VTK-format
# reader lives in pcl/io/vtk_lib_io.h, which needs the VTK libraries —
# the same dependency that keeps pcl/visualization out of this package.
_MESH_FORMATS = {"ply": MESH_FORMAT_PLY, "obj": MESH_FORMAT_OBJ,
                 "vtk": MESH_FORMAT_VTK}
_MESH_READABLE = ("ply", "obj")


cdef _mesh_format(path, format, readable):
    import os.path
    name = format if format is not None else os.path.splitext(str(path))[1][1:]
    name = name.lower()
    allowed = _MESH_READABLE if readable else tuple(sorted(_MESH_FORMATS))
    if name not in allowed:
        raise ValueError(
            "unsupported mesh format %r (%s: %s)"
            % (name, "readable" if readable else "writable",
               ", ".join(allowed)))
    return _MESH_FORMATS[name]


def save_mesh(PointCloud cloud not None, polygons, path, format=None,
              bint binary=False):
    """Write a mesh to *path* (.ply, .obj or .vtk).

    *polygons* is what `reconstruct()` and `reconstruct_with_polygons()`
    return: an iterable of index tuples into *cloud*. So a reconstruction
    can be saved directly:

        mesher = cloud.make_GreedyProjectionTriangulation()
        ...
        pcl.save_mesh(cloud, mesher.reconstruct(), "surface.ply")

    `binary` applies to PLY only; OBJ and VTK are text formats.
    """
    cdef int fmt = _mesh_format(path, format, False)

    cdef vector[int] indices
    cdef vector[int] counts
    cdef Py_ssize_t n = <Py_ssize_t> cloud.size
    cdef int index
    for polygon in polygons:
        counts.push_back(<int> len(polygon))
        for value in polygon:
            index = <int> value
            if index < 0 or index >= n:
                raise IndexError(
                    "polygon index %d is outside the cloud (%d points)"
                    % (index, n))
            indices.push_back(index)

    cdef bytes encoded = str(path).encode()
    cdef string target = encoded
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    cdef int error
    with nogil:
        error = saveMesh(target, deref(src), indices, counts, fmt, binary)
    if error:
        raise IOError("error while saving mesh %s (code %d)" % (path, error))


def load_mesh(path, format=None):
    """Read a mesh from *path* (.ply or .obj).

    Returns ``(cloud, polygons)`` in the same shape `save_mesh` takes and
    `reconstruct_with_polygons()` returns.
    """
    cdef int fmt = _mesh_format(path, format, True)

    cdef shared_ptr[cPointCloud[PointXYZ]] holder
    holder.reset(new cPointCloud[PointXYZ]())
    cdef vector[int] indices
    cdef vector[int] counts
    cdef bytes encoded = str(path).encode()
    cdef string source = encoded
    cdef cPointCloud[PointXYZ]* out = holder.get()
    cdef int error
    with nogil:
        error = loadMesh(source, fmt, deref(out), indices, counts)
    if error:
        raise IOError("error while loading mesh %s (code %d)" % (path, error))

    # Plain loops, not comprehensions: see the Cython traps section of
    # .claude/rules/pipeline.md.
    polygons = []
    cdef Py_ssize_t i, j, offset = 0
    for i in range(<Py_ssize_t> counts.size()):
        polygon = []
        for j in range(counts[i]):
            polygon.append(indices[offset + j])
        offset += counts[i]
        polygons.append(tuple(polygon))
    return wrap_cloud(holder), polygons
