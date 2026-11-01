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
    MESH_FORMAT_IFS, MESH_FORMAT_OBJ, MESH_FORMAT_PLY, MESH_FORMAT_VTK,
    loadMesh,
    saveMesh)

from pcl._pointcloud cimport PointCloud, wrap_cloud
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointNormal, wrap_pointnormal_cloud)
from pcl.pxd.surface.poisson cimport Poisson as cPoisson
from pcl.pxd.surface.marching_cubes_hoppe cimport (
    MarchingCubesHoppe as cMarchingCubesHoppe)
from pcl.pxd.surface.marching_cubes_rbf cimport (
    MarchingCubesRBF as cMarchingCubesRBF)
from pcl.pxd.surface.organized_fast_mesh cimport (
    OrganizedFastMesh as cOrganizedFastMesh)
from pcl.pxd.surface.grid_projection cimport (
    GridProjection as cGridProjection)
from pcl.pxd.surface.bilateral_upsampling cimport (
    BilateralUpsampling as cBilateralUpsampling)
from pcl.pxd.surface.surfel_smoothing cimport (
    SurfelSmoothing as cSurfelSmoothing)
from pcl.pxd.point_types cimport PointXYZRGBA
from pcl.pxd.compat.texture_args cimport textureWithCameras
from pcl.pxd.compat.surface_args cimport (
    ORGANIZED_MESH_QUAD_MESH, ORGANIZED_MESH_TRIANGLE_ADAPTIVE_CUT,
    ORGANIZED_MESH_TRIANGLE_LEFT_CUT, ORGANIZED_MESH_TRIANGLE_RIGHT_CUT,
    earClipping, removeUnusedVertices, setBilateralProjection,
    setOrganizedMeshTriangulation)
from pcl._pointtypes cimport (
    PointCloud_PointXYZRGBA, wrap_normal_cloud, wrap_xyzrgba_cloud)


# OrganizedFastMesh::TriangulationType, re-exported through the shim so
# the values cannot drift from the header.
TRIANGLE_RIGHT_CUT = ORGANIZED_MESH_TRIANGLE_RIGHT_CUT
TRIANGLE_LEFT_CUT = ORGANIZED_MESH_TRIANGLE_LEFT_CUT
TRIANGLE_ADAPTIVE_CUT = ORGANIZED_MESH_TRIANGLE_ADAPTIVE_CUT
QUAD_MESH = ORGANIZED_MESH_QUAD_MESH


cdef _flatten_polygons(polygons, Py_ssize_t n_points, vector[int]& indices,
                       vector[int]& counts):
    """An iterable of index tuples as the flat (indices, counts) pair the
    mesh shims take, refusing indices outside the cloud."""
    cdef int index
    indices.clear()
    counts.clear()
    for polygon in polygons:
        counts.push_back(<int> len(polygon))
        for value in polygon:
            index = <int> value
            if index < 0 or index >= n_points:
                raise IndexError(
                    "polygon index %d is outside the cloud (%d points)"
                    % (index, n_points))
            indices.push_back(index)


cdef object _flat_polygon_list(vector[int]& indices, vector[int]& counts):
    """The flat (indices, counts) pair as a list of index tuples (plain
    loops: see the Cython traps section of .claude/rules/pipeline.md)."""
    polygons = []
    cdef Py_ssize_t i, j, offset = 0
    for i in range(<Py_ssize_t> counts.size()):
        polygon = []
        for j in range(counts[i]):
            polygon.append(indices[offset + j])
        offset += counts[i]
        polygons.append(tuple(polygon))
    return polygons


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


cdef PointCloud_PointNormal _as_pointnormal(cloud, PointCloud_Normal normals):
    """The oriented input the grid reconstructions share: a
    PointCloud_PointNormal as is, or a PointCloud joined with its
    normals. Refuses the shapes that cannot work."""
    cdef PointCloud_PointNormal joined
    if isinstance(cloud, PointCloud_PointNormal):
        if normals is not None:
            raise TypeError(
                "a PointCloud_PointNormal already carries its normals")
        joined = <PointCloud_PointNormal> cloud
    elif isinstance(cloud, PointCloud):
        if normals is None:
            raise TypeError(
                "a PointCloud needs its normals too: "
                "set_InputCloud(cloud, normals)")
        joined = PointCloud_PointNormal().from_xyz_and_normals(
            <PointCloud> cloud, normals)
    else:
        raise TypeError(
            "expected a PointCloud_PointNormal or a PointCloud, got %s"
            % type(cloud).__name__)
    if joined.size == 0:
        raise ValueError("the input cloud is empty")
    return joined


cdef class MarchingCubesHoppe:
    """Marching cubes over Hoppe's signed distance
    (pcl::MarchingCubesHoppe): each grid cell takes its distance to the
    nearest point's tangent plane, so the surface follows the normals
    closely and the cost is one neighbour search per cell.

    Like Poisson it produces a mesh with its OWN vertices, sampled on a
    regular grid: `set_GridResolution(nx, ny, nz)` sets how fine, and
    `set_PercentageExtendGrid` how far past the cloud's bounding box it
    reaches. Takes a :class:`PointCloud_PointNormal`, or a cloud plus
    the normals computed for it:

        mc = cloud.make_MarchingCubesHoppe()
        mc.set_InputCloud(cloud, normals.compute_cloud())
        mc.set_GridResolution(32, 32, 32)
        vertices, triangles = mc.reconstruct()
    """

    cdef cMarchingCubesHoppe[PointNormal]* me
    cdef bint has_input

    def __cinit__(self, pc=None):
        self.me = new cMarchingCubesHoppe[PointNormal]()
        self.has_input = False
        if isinstance(pc, PointCloud_PointNormal):
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, cloud not None, PointCloud_Normal normals=None):
        """Take a :class:`PointCloud_PointNormal`, or a :class:`PointCloud`
        together with its normals."""
        cdef PointCloud_PointNormal joined = _as_pointnormal(cloud, normals)
        self.me.setInputCloud(joined.thisptr_shared)
        self.has_input = True

    def set_IsoLevel(self, float level):
        """Implicit-function value the surface is drawn at (0 = on the
        points)."""
        self.me.setIsoLevel(level)

    def get_IsoLevel(self):
        return self.me.getIsoLevel()

    def set_GridResolution(self, int nx, int ny, int nz):
        """Grid cells along each axis (PCL's default is 32 each)."""
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("grid resolution must be at least 1 per axis")
        self.me.setGridResolution(nx, ny, nz)

    def set_PercentageExtendGrid(self, float percentage):
        """Margin around the bounding box, as a fraction of its size."""
        self.me.setPercentageExtendGrid(percentage)

    def get_PercentageExtendGrid(self):
        return self.me.getPercentageExtendGrid()

    def set_DistanceIgnore(self, float distance):
        """Grid cells farther than this from any point are left empty
        (negative, the default, means no limit)."""
        self.me.setDistanceIgnore(distance)

    def get_DistanceIgnore(self):
        return self.me.getDistanceIgnore()

    def reconstruct(self):
        """Return ``(vertices, triangles)``: a :class:`PointCloud_PointNormal`
        of the mesh's own vertices and index triples into it."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before reconstruct()")
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = holder.get()
        cdef vector[Vertices] polygons
        cdef cMarchingCubesHoppe[PointNormal]* mesher = self.me
        with nogil:
            mesher.reconstruct(deref(out), polygons)
        return wrap_pointnormal_cloud(holder), _polygon_list(polygons)


cdef class MarchingCubesRBF:
    """Marching cubes over a radial-basis-function surface
    (pcl::MarchingCubesRBF): an implicit function is fitted through every
    point and an off-surface copy of it, which solves a dense system in
    the number of points -- keep the cloud small (hundreds of points).

    Like Poisson it produces a mesh with its OWN vertices, sampled on a
    regular grid: `set_GridResolution(nx, ny, nz)` sets how fine, and
    `set_PercentageExtendGrid` how far past the cloud's bounding box it
    reaches. Takes a :class:`PointCloud_PointNormal`, or a cloud plus
    the normals computed for it:

        mc = cloud.make_MarchingCubesRBF()
        mc.set_InputCloud(cloud, normals.compute_cloud())
        mc.set_GridResolution(32, 32, 32)
        vertices, triangles = mc.reconstruct()
    """

    cdef cMarchingCubesRBF[PointNormal]* me
    cdef bint has_input

    def __cinit__(self, pc=None):
        self.me = new cMarchingCubesRBF[PointNormal]()
        self.has_input = False
        if isinstance(pc, PointCloud_PointNormal):
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, cloud not None, PointCloud_Normal normals=None):
        """Take a :class:`PointCloud_PointNormal`, or a :class:`PointCloud`
        together with its normals."""
        cdef PointCloud_PointNormal joined = _as_pointnormal(cloud, normals)
        self.me.setInputCloud(joined.thisptr_shared)
        self.has_input = True

    def set_IsoLevel(self, float level):
        """Implicit-function value the surface is drawn at (0 = on the
        points)."""
        self.me.setIsoLevel(level)

    def get_IsoLevel(self):
        return self.me.getIsoLevel()

    def set_GridResolution(self, int nx, int ny, int nz):
        """Grid cells along each axis (PCL's default is 32 each)."""
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("grid resolution must be at least 1 per axis")
        self.me.setGridResolution(nx, ny, nz)

    def set_PercentageExtendGrid(self, float percentage):
        """Margin around the bounding box, as a fraction of its size."""
        self.me.setPercentageExtendGrid(percentage)

    def get_PercentageExtendGrid(self):
        return self.me.getPercentageExtendGrid()

    def set_OffSurfaceDisplacement(self, float epsilon):
        """How far along each normal the off-surface constraint points
        are placed (PCL's default 0.1)."""
        self.me.setOffSurfaceDisplacement(epsilon)

    def get_OffSurfaceDisplacement(self):
        return self.me.getOffSurfaceDisplacement()

    def reconstruct(self):
        """Return ``(vertices, triangles)``: a :class:`PointCloud_PointNormal`
        of the mesh's own vertices and index triples into it."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before reconstruct()")
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = holder.get()
        cdef vector[Vertices] polygons
        cdef cMarchingCubesRBF[PointNormal]* mesher = self.me
        with nogil:
            mesher.reconstruct(deref(out), polygons)
        return wrap_pointnormal_cloud(holder), _polygon_list(polygons)


cdef class Poisson:
    """Watertight surface from an oriented cloud (pcl::Poisson).

    Where GreedyProjectionTriangulation connects the input points,
    Poisson fits an implicit surface to them and extracts a NEW mesh from
    it, so `reconstruct()` returns its own vertices as well as the
    polygons. Takes a :class:`PointCloud_PointNormal`, or a cloud plus
    the normals computed for it:

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        poisson = cloud.make_Poisson()
        poisson.set_InputCloud(cloud, normals.compute_cloud())
        poisson.set_Depth(8)
        vertices, triangles = poisson.reconstruct()
        pcl.save_mesh(vertices.to_xyz_cloud(), triangles, "surface.ply")

    The normals must be consistently oriented (all outward or all
    inward); the surface follows them.
    """

    cdef cPoisson[PointNormal]* me
    cdef bint has_input

    def __cinit__(self, pc=None):
        self.me = new cPoisson[PointNormal]()
        self.has_input = False
        # A plain PointCloud alone is not enough input (no normals), so
        # only a PointCloud_PointNormal is taken here; set_InputCloud
        # does the real work otherwise.
        if isinstance(pc, PointCloud_PointNormal):
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, cloud not None, PointCloud_Normal normals=None):
        """Take a :class:`PointCloud_PointNormal`, or a :class:`PointCloud`
        together with its normals."""
        cdef PointCloud_PointNormal joined = _as_pointnormal(cloud, normals)
        self.me.setInputCloud(joined.thisptr_shared)
        self.has_input = True

    def set_Depth(self, int depth):
        """Octree depth bounding the resolution (2^depth cells per axis);
        PCL's default is 8."""
        self.me.setDepth(depth)

    def get_Depth(self):
        return self.me.getDepth()

    def set_MinDepth(self, int depth):
        self.me.setMinDepth(depth)

    def get_MinDepth(self):
        return self.me.getMinDepth()

    def set_PointWeight(self, float weight):
        """How much the surface is pulled towards the samples (0 is the
        original, unconstrained Poisson fit)."""
        self.me.setPointWeight(weight)

    def get_PointWeight(self):
        return self.me.getPointWeight()

    def set_Scale(self, float scale):
        self.me.setScale(scale)

    def get_Scale(self):
        return self.me.getScale()

    def set_SolverDivide(self, int depth):
        self.me.setSolverDivide(depth)

    def get_SolverDivide(self):
        return self.me.getSolverDivide()

    def set_IsoDivide(self, int depth):
        self.me.setIsoDivide(depth)

    def get_IsoDivide(self):
        return self.me.getIsoDivide()

    def set_SamplesPerNode(self, float samples):
        """Raise (15-20) for noisy clouds to smooth the result."""
        self.me.setSamplesPerNode(samples)

    def get_SamplesPerNode(self):
        return self.me.getSamplesPerNode()

    def set_Confidence(self, bint confidence):
        """Use the normals' lengths as confidence instead of normalising
        them."""
        self.me.setConfidence(confidence)

    def get_Confidence(self):
        return self.me.getConfidence()

    def set_OutputPolygons(self, bint polygons):
        """Return the marching-cubes polygons as they are instead of
        triangulating them."""
        self.me.setOutputPolygons(polygons)

    def get_OutputPolygons(self):
        return self.me.getOutputPolygons()

    def set_Degree(self, int degree):
        self.me.setDegree(degree)

    def get_Degree(self):
        return self.me.getDegree()

    def set_Manifold(self, bint manifold):
        self.me.setManifold(manifold)

    def get_Manifold(self):
        return self.me.getManifold()

    def reconstruct(self):
        """Return ``(vertices, polygons)``: a :class:`PointCloud_PointNormal`
        of the mesh's own vertices and index tuples into it (triangles
        unless `set_OutputPolygons(True)`)."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before reconstruct()")
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = holder.get()
        cdef vector[Vertices] polygons
        cdef cPoisson[PointNormal]* mesher = self.me
        with nogil:
            mesher.reconstruct(deref(out), polygons)
        return wrap_pointnormal_cloud(holder), _polygon_list(polygons)


# Mesh file formats, by extension. VTK is write-only: PCL's VTK-format
# reader lives in pcl/io/vtk_lib_io.h, which needs the VTK libraries —
# the same dependency that keeps pcl/visualization out of this package.
_MESH_FORMATS = {"ply": MESH_FORMAT_PLY, "obj": MESH_FORMAT_OBJ,
                 "vtk": MESH_FORMAT_VTK, "ifs": MESH_FORMAT_IFS}
_MESH_READABLE = ("ifs", "obj", "ply")
_MESH_WRITABLE = ("obj", "ply", "vtk")
# What only PCL's VTK-based I/O handles: pcl._vtk_mesh, built where PCL
# has VTK.
_VTK_MESH_READABLE = ("stl", "vtk")
_VTK_MESH_WRITABLE = ("stl",)


def texture_mesh_with_cameras(PointCloud cloud not None, triangles, cameras):
    """Which camera textures which face, and where
    (pcl::TextureMapping::textureMeshwithMultipleCameras).

    `triangles` is (m, 3) vertex indices. Each camera is a dict with
    "pose" (4 x 4, camera -> world; the camera looks along +z, x right,
    y down), "focal_length" (or "focal_length_w" / "focal_length_h"),
    "width" and "height" in pixels, and optionally "center" (cx, cy),
    default the image centre. Every camera, in order, takes the faces it
    sees unoccluded among those the cameras before it left.

    Returns one ``(faces, uv)`` pair per camera plus a last pair for the
    faces no camera saw: ``faces`` is (k, 3) vertex indices, ``uv`` is
    (k, 3, 2) texture coordinates of those vertices -- u to the right and
    v UP, both 0..1 over the image (PCL's convention), -1 for the unseen
    group. A principal point of exactly 0 counts as unset (PCL tests
    ``center > 0``)."""
    import numpy as np
    tri = np.asarray(triangles)
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise ValueError("triangles must be a non-empty (m, 3) array")
    if tri.min() < 0 or tri.max() >= cloud.size:
        raise ValueError("triangle indices must lie in 0..%d" % (cloud.size - 1))
    if not cameras:
        raise ValueError("at least one camera is needed")
    flat = []
    for cam in cameras:
        pose = np.asarray(cam["pose"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("a camera pose is 4 x 4")
        fw = float(cam.get("focal_length_w", cam.get("focal_length", 0)))
        fh = float(cam.get("focal_length_h", cam.get("focal_length", 0)))
        width, height = float(cam["width"]), float(cam["height"])
        if not (fw > 0 and fh > 0 and width > 0 and height > 0):
            raise ValueError("focal lengths, width and height must be > 0")
        cx, cy = cam.get("center", (width / 2.0, height / 2.0))
        flat.extend(pose.ravel(order="F").tolist())
        flat.extend([fw, fh, float(cx), float(cy), width, height])
    cdef vector[int] vtri = np.ascontiguousarray(tri, dtype=np.int32).ravel().tolist()
    cdef vector[double] vcams = flat
    cdef vector[int] sizes
    cdef vector[int] faces
    cdef vector[float] uv
    with nogil:
        textureWithCameras(cloud.thisptr_shared, vtri, vcams, sizes, faces, uv)
    all_faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    all_uv = np.asarray(uv, dtype=np.float32).reshape(-1, 3, 2)
    out = []
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t> sizes.size()):
        out.append((all_faces[start:start + sizes[i]],
                    all_uv[start:start + sizes[i]]))
        start += sizes[i]
    return out


cdef str _mesh_name(path, format):
    import os.path
    name = format if format is not None else os.path.splitext(str(path))[1][1:]
    return name.lower()


cdef _mesh_format(path, format, readable):
    name = _mesh_name(path, format)
    allowed = _MESH_READABLE if readable else _MESH_WRITABLE
    if name not in allowed:
        vtk = _VTK_MESH_READABLE if readable else _VTK_MESH_WRITABLE
        raise ValueError(
            "unsupported mesh format %r (%s: %s, and %s where PCL has VTK)"
            % (name, "readable" if readable else "writable",
               ", ".join(allowed), ", ".join(vtk)))
    return _MESH_FORMATS[name]


cdef object _vtk_mesh_module(str name):
    try:
        from pcl import _vtk_mesh
    except ImportError:
        raise ValueError("%s meshes need PCL built with VTK (pcl._vtk_mesh "
                         "is not built here)" % name) from None
    return _vtk_mesh


def save_mesh(PointCloud cloud not None, polygons, path, format=None,
              bint binary=False):
    """Write a mesh to *path* (.ply, .obj or .vtk).

    *polygons* is what `reconstruct()` and `reconstruct_with_polygons()`
    return: an iterable of index tuples into *cloud*. So a reconstruction
    can be saved directly:

        mesher = cloud.make_GreedyProjectionTriangulation()
        ...
        pcl.save_mesh(cloud, mesher.reconstruct(), "surface.ply")

    `binary` applies to PLY (and STL); OBJ and VTK are text formats.
    STL goes through PCL's VTK-based writer, so it needs a PCL built
    with VTK (`pcl.save_polygon_file`).
    """
    name = _mesh_name(path, format)
    if name in _VTK_MESH_WRITABLE:
        _vtk_mesh_module(name).save_polygon_file(path, cloud, polygons, binary)
        return
    cdef int fmt = _mesh_format(path, format, False)

    cdef vector[int] indices
    cdef vector[int] counts
    _flatten_polygons(polygons, cloud.size, indices, counts)

    cdef bytes encoded = str(path).encode()
    cdef string target = encoded
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    cdef int error
    with nogil:
        error = saveMesh(target, deref(src), indices, counts, fmt, binary)
    if error:
        raise IOError("error while saving mesh %s (code %d)" % (path, error))


def load_mesh(path, format=None):
    """Read a mesh from *path* (.ply, .obj or .ifs; .stl and .vtk where
    PCL has VTK, through `pcl.load_polygon_file`).

    Returns ``(cloud, polygons)`` in the same shape `save_mesh` takes and
    `reconstruct_with_polygons()` returns.
    """
    name = _mesh_name(path, format)
    if name in _VTK_MESH_READABLE:
        return _vtk_mesh_module(name).load_polygon_file(path)
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

    return wrap_cloud(holder), _flat_polygon_list(indices, counts)


cdef class OrganizedFastMesh:
    """Mesh an organized cloud by connecting neighbouring pixels
    (pcl::OrganizedFastMesh).

    No neighbour search, so it runs at sensor frame rate; the price is
    that it needs the pixel grid (`PointCloud.from_organized_array`). The
    polygons index the INPUT cloud, NaN pixels included — save them with
    `pcl.save_mesh(cloud, polygons, path)`. `set_TriangulationType` takes
    `pcl.TRIANGLE_RIGHT_CUT`, `TRIANGLE_LEFT_CUT`, `TRIANGLE_ADAPTIVE_CUT`
    or `QUAD_MESH` (PCL's default, four indices per face).
    """

    cdef cOrganizedFastMesh[PointXYZ]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cOrganizedFastMesh[PointXYZ]()
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if pc.ptr().height < 2:
            raise ValueError(
                "OrganizedFastMesh needs an organized cloud (height > 1); "
                "build one with PointCloud.from_organized_array")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_TriangulationType(self, int kind):
        if kind not in (TRIANGLE_RIGHT_CUT, TRIANGLE_LEFT_CUT,
                        TRIANGLE_ADAPTIVE_CUT, QUAD_MESH):
            raise ValueError("unknown triangulation type %r" % kind)
        setOrganizedMeshTriangulation(deref(self.me), kind)

    def set_TrianglePixelSize(self, int size):
        """Pixels along a face's edge, both end pixels counted, on both
        axes. PCL stores ``max(1, size - 1)`` as the step, so 1 (the
        default) and 2 both connect neighbouring pixels and 3 is the
        first value that skips one."""
        if size < 1:
            raise ValueError("triangle pixel size must be >= 1")
        self.me.setTrianglePixelSize(size)

    def set_TrianglePixelSizeRows(self, int size):
        """`set_TrianglePixelSize` for the vertical edges only."""
        if size < 1:
            raise ValueError("triangle pixel size must be >= 1")
        self.me.setTrianglePixelSizeRows(size)

    def set_TrianglePixelSizeColumns(self, int size):
        """`set_TrianglePixelSize` for the horizontal edges only."""
        if size < 1:
            raise ValueError("triangle pixel size must be >= 1")
        self.me.setTrianglePixelSizeColumns(size)

    def set_MaxEdgeLength(self, float a, float b=0.0, float c=0.0):
        """Drop faces with an edge longer than ``a + b*z + c*z**2``
        (``z`` the depth), which keeps depth discontinuities open."""
        self.me.setMaxEdgeLength(a, b, c)

    def unset_MaxEdgeLength(self):
        self.me.unsetMaxEdgeLength()

    def store_ShadowedFaces(self, bint enable):
        """Keep faces seen at a grazing angle from the viewpoint (dropped
        by default)."""
        self.me.storeShadowedFaces(enable)

    def set_AngleTolerance(self, float radians):
        """Grazing angle below which a face counts as shadowed."""
        self.me.setAngleTolerance(radians)

    def set_DistanceTolerance(self, float distance,
                              bint depth_dependent=False):
        """Distance a shadowed face may still span."""
        self.me.setDistanceTolerance(distance, depth_dependent)

    def use_DepthAsDistance(self, bint enable):
        """Use z instead of the distance to the viewpoint (faster, and
        what an RGB-D camera means by depth)."""
        self.me.useDepthAsDistance(enable)

    def reconstruct(self):
        """Return the faces as a list of index tuples into the input."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before reconstruct()")
        cdef vector[Vertices] polygons
        cdef cOrganizedFastMesh[PointXYZ]* mesher = self.me
        with nogil:
            mesher.reconstruct(polygons)
        return _polygon_list(polygons)


cdef class GridProjection:
    """Surface reconstruction by grid projection (pcl::GridProjection):
    the corners of every grid cell near the surface are projected onto
    it, giving a watertight mesh with its OWN vertices.

    `set_Resolution` is the cell size in the cloud's own units and is the
    setting that matters: PCL's default, 0.001, is sized for a cloud in
    metres captured at millimetre detail and turns a unit-scale cloud
    into millions of cells. Takes a :class:`PointCloud_PointNormal`, or a
    cloud plus its normals.
    """

    cdef cGridProjection[PointNormal]* me
    cdef bint has_input

    def __cinit__(self, pc=None):
        self.me = new cGridProjection[PointNormal]()
        self.has_input = False
        if isinstance(pc, PointCloud_PointNormal):
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, cloud not None, PointCloud_Normal normals=None):
        """Take a :class:`PointCloud_PointNormal`, or a :class:`PointCloud`
        together with its normals."""
        cdef PointCloud_PointNormal joined = _as_pointnormal(cloud, normals)
        self.me.setInputCloud(joined.thisptr_shared)
        self.has_input = True

    def set_Resolution(self, double resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.me.setResolution(resolution)

    def get_Resolution(self):
        return self.me.getResolution()

    def set_PaddingSize(self, int cells):
        """Cells around each occupied one that are also evaluated."""
        if cells < 1:
            raise ValueError("padding size must be >= 1")
        self.me.setPaddingSize(cells)

    def get_PaddingSize(self):
        return self.me.getPaddingSize()

    def set_NearestNeighborNum(self, int k):
        """Neighbours used for the local surface at each corner."""
        if k < 1:
            raise ValueError("k must be >= 1")
        self.me.setNearestNeighborNum(k)

    def get_NearestNeighborNum(self):
        return self.me.getNearestNeighborNum()

    def set_MaxBinarySearchLevel(self, int levels):
        """Bisection steps when projecting a corner onto the surface."""
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.me.setMaxBinarySearchLevel(levels)

    def get_MaxBinarySearchLevel(self):
        return self.me.getMaxBinarySearchLevel()

    def reconstruct(self):
        """Return ``(vertices, polygons)``: a :class:`PointCloud_PointNormal`
        of the mesh's own vertices and index tuples into it."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before reconstruct()")
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = holder.get()
        cdef vector[Vertices] polygons
        cdef cGridProjection[PointNormal]* mesher = self.me
        with nogil:
            mesher.reconstruct(deref(out), polygons)
        return wrap_pointnormal_cloud(holder), _polygon_list(polygons)


cdef float[9] _KINECT_VGA = [525.0, 0.0, 320.0,
                             0.0, 525.0, 240.0,
                             0.0, 0.0, 1.0]


cdef class BilateralUpsampling:
    """Fill the depth holes of an organized RGB-D frame
    (pcl::BilateralUpsampling): each pixel's depth becomes a bilateral
    average of its window, weighted by distance in the image AND by
    colour similarity, so edges in the colour image stay edges in depth.
    Every pixel is then re-projected through the camera matrix.

    Input and output are :class:`PointCloud_PointXYZRGBA` with the same
    grid (`from_organized_rgba_array`). `set_ProjectionMatrix` takes the
    3x3 camera matrix; it defaults to the Kinect VGA intrinsics PCL ships
    (f = 525, centre 320, 240) because PCL itself leaves the matrix
    uninitialised. PCL prints the inverted matrix to stdout on every
    `process()` call; that is PCL's, not this wrapper's.
    """

    cdef cBilateralUpsampling[PointXYZRGBA, PointXYZRGBA]* me
    cdef bint has_input

    def __cinit__(self, PointCloud_PointXYZRGBA pc=None):
        self.me = new cBilateralUpsampling[PointXYZRGBA, PointXYZRGBA]()
        setBilateralProjection(deref(self.me), _KINECT_VGA)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        if pc.ptr().height < 2:
            raise ValueError(
                "BilateralUpsampling needs an organized cloud (height > 1); "
                "build one with from_organized_rgba_array")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_ProjectionMatrix(self, matrix):
        """The 3x3 camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``."""
        import numpy as np
        arr = np.ascontiguousarray(matrix, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError("projection matrix must be 3x3, got %s"
                             % (arr.shape,))
        if abs(float(np.linalg.det(arr))) < 1e-12:
            raise ValueError("projection matrix must be invertible")
        cdef float buf[9]
        cdef Py_ssize_t i
        flat = arr.reshape(9)
        for i in range(9):
            buf[i] = flat[i]
        setBilateralProjection(deref(self.me), buf)

    def set_WindowSize(self, int size):
        """Half-width of the averaging window, in pixels."""
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.me.setWindowSize(size)

    def get_WindowSize(self):
        return self.me.getWindowSize()

    def set_SigmaColor(self, float sigma):
        """Colour difference (sum of |dr|+|dg|+|db|) a neighbour may have
        and still count."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.me.setSigmaColor(sigma)

    def get_SigmaColor(self):
        return self.me.getSigmaColor()

    def set_SigmaDepth(self, float sigma):
        """Spatial falloff of the window, in pixels."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.me.setSigmaDepth(sigma)

    def get_SigmaDepth(self):
        return self.me.getSigmaDepth()

    def process(self):
        """Return the filled frame as a new organized
        :class:`PointCloud_PointXYZRGBA`."""
        if not self.has_input:
            raise RuntimeError(
                "set_InputCloud(...) is required before process()")
        cdef shared_ptr[cPointCloud[PointXYZRGBA]] holder
        holder.reset(new cPointCloud[PointXYZRGBA]())
        cdef cPointCloud[PointXYZRGBA]* out = holder.get()
        cdef cBilateralUpsampling[PointXYZRGBA, PointXYZRGBA]* f = self.me
        with nogil:
            f.process(deref(out))
        return wrap_xyzrgba_cloud(holder)


cdef class SurfelSmoothing:
    """Smooth points and normals together (pcl::SurfelSmoothing): every
    surfel moves towards the tangent planes of its neighbours, weighted
    by a Gaussian of width `scale` (in the cloud's own units).

        smoother = pcl.SurfelSmoothing(scale=0.05)
        points, normals = smoother.compute(cloud, normals)
    """

    cdef cSurfelSmoothing[PointXYZ, Normal]* me
    cdef float _scale

    def __cinit__(self, float scale=0.01):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._scale = scale
        self.me = new cSurfelSmoothing[PointXYZ, Normal](scale)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    @property
    def scale(self):
        return self._scale

    def compute(self, PointCloud cloud not None,
                PointCloud_Normal normals not None):
        """Return ``(points, normals)``: a smoothed :class:`PointCloud` and
        its :class:`PointCloud_Normal`, one entry per input point.

        PCL returns nothing at all (and only logs) when the normals are
        missing or do not match the cloud; this raises instead.
        """
        if cloud.size == 0:
            raise ValueError("the input cloud is empty")
        if normals.size != cloud.size:
            raise ValueError(
                "normals must match the cloud: %d points, %d normals"
                % (cloud.size, normals.size))
        cdef shared_ptr[cPointCloud[PointXYZ]] out_points
        cdef shared_ptr[cPointCloud[Normal]] out_normals
        out_points.reset(new cPointCloud[PointXYZ]())
        out_normals.reset(new cPointCloud[Normal]())
        cdef shared_ptr[cPointCloud[Normal]] in_normals = normals.thisptr_shared
        self.me.setInputCloud(cloud.thisptr_shared)
        self.me.setInputNormals(in_normals)
        with nogil:
            self.me.computeSmoothedCloud(out_points, out_normals)
        return wrap_cloud(out_points), wrap_normal_cloud(out_normals)


cdef class EarClipping:
    """Triangulate simple planar polygons by ear clipping
    (pcl::EarClipping): a polygon of n vertices becomes n-2 triangles
    over the same vertices. Triangles pass through unchanged.

        triangles = pcl.EarClipping().process(cloud, polygons)

    PCL 1.14's implementation is correct on every convex polygon but
    mis-triangulates most concave ones — measured on L, U and C shapes in
    both windings, it returns triangles that cover area outside the
    polygon. Each result is therefore checked, and a polygon PCL got
    wrong raises RuntimeError instead of coming back as a broken mesh.
    """

    def process(self, PointCloud cloud not None, polygons):
        """Return the triangles as a list of index triples into *cloud*."""
        cdef vector[int] indices, counts, out_indices, out_counts
        _flatten_polygons(polygons, cloud.size, indices, counts)
        cdef size_t i
        for i in range(counts.size()):
            if counts[i] < 3:
                raise ValueError(
                    "polygon %d has %d vertices; ear clipping needs at "
                    "least 3" % (i, counts[i]))
        cdef cPointCloud[PointXYZ]* src = cloud.ptr()
        cdef int failed
        with nogil:
            failed = earClipping(deref(src), indices, counts, out_indices,
                                 out_counts)
        if failed >= 0:
            raise RuntimeError(
                "PCL's EarClipping mis-triangulated polygon %d: its "
                "triangles cover area outside the polygon. PCL 1.14 handles "
                "convex polygons but gets most concave ones wrong" % failed)
        return _flat_polygon_list(out_indices, out_counts)


cdef class SimplificationRemoveUnusedVertices:
    """Drop the vertices no polygon uses and renumber the rest
    (pcl::surface::SimplificationRemoveUnusedVertices) — what a mesh
    cropped down to a few faces needs before saving."""

    def simplify(self, PointCloud cloud not None, polygons):
        """Return ``(vertices, polygons, kept)``: the compacted cloud, the
        polygons renumbered into it, and for each kept vertex its index
        in the original *cloud*."""
        import numpy as np
        cdef vector[int] indices, counts, out_indices, out_counts, kept
        _flatten_polygons(polygons, cloud.size, indices, counts)
        cdef shared_ptr[cPointCloud[PointXYZ]] holder
        holder.reset(new cPointCloud[PointXYZ]())
        cdef cPointCloud[PointXYZ]* src = cloud.ptr()
        cdef cPointCloud[PointXYZ]* out = holder.get()
        with nogil:
            removeUnusedVertices(deref(src), indices, counts, deref(out),
                                 out_indices, out_counts, kept)
        kept_array = np.empty(kept.size(), dtype=np.int32)
        cdef int[::1] view = kept_array
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> kept.size()):
            view[i] = kept[i]
        return (wrap_cloud(holder), _flat_polygon_list(out_indices, out_counts),
                kept_array)
