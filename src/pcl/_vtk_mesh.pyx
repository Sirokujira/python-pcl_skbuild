# distutils: language = c++
# cython: language_level=3
"""PCL's VTK mesh code: smoothing, subdivision and decimation
(pcl/surface/vtk_smoothing) and STL / VTK / PLY / OBJ mesh files
(pcl/io/vtk_lib_io.h).

Built only where PCL was built with VTK (it installs these headers only
then, and they include VTK's); the four classes and the two functions
are absent otherwise, and `pcl.save_mesh` / `pcl.load_mesh` then refuse
.stl. A mesh is a `PointCloud` plus a list of vertex-index tuples, as
for `pcl.load_mesh` / `pcl.save_mesh`; `process` returns a new
``(cloud, polygons)`` pair. Polygons are triangulated on the way in (PCL
passes every mesh through vtkTriangleFilter), so the output is always
triangles. Goes through pcl/compat/vtk_mesh_args.h, which also says
which two of PCL's implementations are replaced and why.
"""

from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.compat.vtk_mesh_args cimport (
    LaplacianSmoothing, QuadricDecimation, Subdivision, WindowedSincSmoothing,
    loadPolygonFileFlat, savePolygonFileFlat)

from pcl._pointcloud cimport PointCloud, wrap_cloud


ctypedef fused MeshProcessor:
    LaplacianSmoothing
    WindowedSincSmoothing
    Subdivision
    QuadricDecimation


cdef _flatten(polygons, Py_ssize_t n_points, vector[int]& indices,
              vector[int]& counts):
    """Index tuples as the flat (indices, counts) pair, refusing what PCL
    would crash on (no faces, an empty cloud) or misread."""
    cdef int index
    indices.clear()
    counts.clear()
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("a polygon needs at least 3 vertices, got %d"
                             % len(polygon))
        counts.push_back(<int> len(polygon))
        for value in polygon:
            index = <int> value
            if index < 0 or index >= n_points:
                raise IndexError(
                    "polygon index %d is outside the cloud (%d points)"
                    % (index, n_points))
            indices.push_back(index)
    if counts.empty():
        raise ValueError("the mesh has no polygons")


cdef object _polygons(vector[int]& indices, vector[int]& counts):
    """The flat pair as index tuples (plain loops: see the Cython traps in
    .claude/rules/pipeline.md)."""
    polygons = []
    cdef Py_ssize_t i, j, offset = 0
    for i in range(<Py_ssize_t> counts.size()):
        polygon = []
        for j in range(counts[i]):
            polygon.append(indices[offset + j])
        offset += counts[i]
        polygons.append(tuple(polygon))
    return polygons


cdef object _process(MeshProcessor* processor, PointCloud cloud, polygons):
    """``(cloud, polygons)``, or None when the processor returned no mesh."""
    cdef vector[int] indices, counts, out_indices, out_counts
    _flatten(polygons, cloud.size, indices, counts)
    cdef shared_ptr[cPointCloud[PointXYZ]] holder
    holder.reset(new cPointCloud[PointXYZ]())
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    cdef cPointCloud[PointXYZ]* out = holder.get()
    cdef bint ok
    with nogil:
        ok = processor.processFlat(deref(src), indices, counts, deref(out),
                                   out_indices, out_counts)
    if not ok:
        return None
    return wrap_cloud(holder), _polygons(out_indices, out_counts)


cdef _angle(double degrees, str what):
    if not 0 <= degrees <= 180:
        raise ValueError("%s must be in 0..180 degrees" % what)


cdef class MeshSmoothingLaplacianVTK:
    """Laplacian smoothing (pcl::MeshSmoothingLaplacianVTK, VTK's
    vtkSmoothPolyDataFilter): every iteration moves each vertex
    `relaxation_factor` of the way towards the mean of its neighbours.

        cloud, polygons = pcl.MeshSmoothingLaplacianVTK(num_iter=50) \\
            .process(cloud, polygons)

    Defaults are PCL's. `convergence` > 0 stops early once no vertex
    moves more than that fraction of the mesh's bounding-box diagonal;
    `feature_edge_smoothing` keeps edges sharper than `feature_angle`
    (degrees) from being smoothed across; `boundary_smoothing=False`
    pins the boundary vertices. Vertices keep their order. Laplacian
    smoothing shrinks a closed surface as it smooths;
    `MeshSmoothingWindowedSincVTK` does not.
    """

    cdef LaplacianSmoothing* me

    def __cinit__(self, int num_iter=20, double convergence=0.0,
                  double relaxation_factor=0.01,
                  bint feature_edge_smoothing=False,
                  double feature_angle=45.0, double edge_angle=15.0,
                  bint boundary_smoothing=True):
        self.me = new LaplacianSmoothing()
        self.set_NumIter(num_iter)
        self.set_Convergence(convergence)
        self.set_RelaxationFactor(relaxation_factor)
        self.set_FeatureEdgeSmoothing(feature_edge_smoothing)
        self.set_FeatureAngle(feature_angle)
        self.set_EdgeAngle(edge_angle)
        self.set_BoundarySmoothing(boundary_smoothing)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_NumIter(self, int num_iter):
        if num_iter < 0:
            raise ValueError("num_iter must be >= 0")
        self.me.setNumIter(num_iter)

    def set_Convergence(self, double convergence):
        if convergence < 0:
            raise ValueError("convergence must be >= 0")
        self.me.setConvergence(convergence)

    def set_RelaxationFactor(self, double relaxation_factor):
        if relaxation_factor <= 0:
            raise ValueError("relaxation_factor must be > 0")
        self.me.setRelaxationFactor(relaxation_factor)

    def set_FeatureEdgeSmoothing(self, bint value):
        self.me.setFeatureEdgeSmoothing(value)

    def set_FeatureAngle(self, double degrees):
        _angle(degrees, "feature_angle")
        self.me.setFeatureAngle(degrees)

    def set_EdgeAngle(self, double degrees):
        _angle(degrees, "edge_angle")
        self.me.setEdgeAngle(degrees)

    def set_BoundarySmoothing(self, bint value):
        self.me.setBoundarySmoothing(value)

    def process(self, PointCloud cloud not None, polygons):
        result = _process(self.me, cloud, polygons)
        if result is None:
            raise RuntimeError("VTK's Laplacian smoothing returned no mesh")
        return result


cdef class MeshSmoothingWindowedSincVTK:
    """Windowed-sinc smoothing (pcl::MeshSmoothingWindowedSincVTK, VTK's
    vtkWindowedSincPolyDataFilter): a low-pass filter on the mesh, which
    removes noise without the shrinking of Laplacian smoothing.

        cloud, polygons = pcl.MeshSmoothingWindowedSincVTK().process(
            cloud, polygons)

    Defaults are PCL's. `pass_band` (0..2) is the filter's cut-off --
    smaller smooths more; `normalize_coordinates` scales the mesh into a
    unit cube for the computation (numerically safer far from the
    origin). The feature / boundary options are as for
    `MeshSmoothingLaplacianVTK`. Vertices keep their order. PCL's own
    implementation leaks the VTK filter and its output on every call; it
    is replaced (compat/vtk_mesh_args.h).
    """

    cdef WindowedSincSmoothing* me

    def __cinit__(self, int num_iter=20, double pass_band=0.1,
                  bint normalize_coordinates=False,
                  bint feature_edge_smoothing=False,
                  double feature_angle=45.0, double edge_angle=15.0,
                  bint boundary_smoothing=True):
        self.me = new WindowedSincSmoothing()
        self.set_NumIter(num_iter)
        self.set_PassBand(pass_band)
        self.set_NormalizeCoordinates(normalize_coordinates)
        self.set_FeatureEdgeSmoothing(feature_edge_smoothing)
        self.set_FeatureAngle(feature_angle)
        self.set_EdgeAngle(edge_angle)
        self.set_BoundarySmoothing(boundary_smoothing)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_NumIter(self, int num_iter):
        if num_iter < 0:
            raise ValueError("num_iter must be >= 0")
        self.me.setNumIter(num_iter)

    def set_PassBand(self, double pass_band):
        if not 0 < pass_band <= 2:
            raise ValueError("pass_band must be in (0, 2]")
        self.me.setPassBand(pass_band)

    def set_NormalizeCoordinates(self, bint value):
        self.me.setNormalizeCoordinates(value)

    def set_FeatureEdgeSmoothing(self, bint value):
        self.me.setFeatureEdgeSmoothing(value)

    def set_FeatureAngle(self, double degrees):
        _angle(degrees, "feature_angle")
        self.me.setFeatureAngle(degrees)

    def set_EdgeAngle(self, double degrees):
        _angle(degrees, "edge_angle")
        self.me.setEdgeAngle(degrees)

    def set_BoundarySmoothing(self, bint value):
        self.me.setBoundarySmoothing(value)

    def process(self, PointCloud cloud not None, polygons):
        result = _process(self.me, cloud, polygons)
        if result is None:
            raise RuntimeError("VTK's windowed-sinc smoothing returned no mesh")
        return result


_SUBDIVISION_FILTERS = {"linear": 0, "loop": 1, "butterfly": 2}


cdef class MeshSubdivisionVTK:
    """Subdivision (pcl::MeshSubdivisionVTK): every triangle becomes four.

        cloud, polygons = pcl.MeshSubdivisionVTK("loop").process(
            cloud, polygons)

    `filter` is "linear" (new vertices at the edge midpoints, the shape
    unchanged), "loop" (approximating: every vertex moves, the surface
    gets smoother) or "butterfly" (interpolating: the original vertices
    stay put). The original vertices come first in the output, in their
    order. VTK's Loop filter fails on a mesh with a vertex no face uses
    -- PCL then segfaults; here it is a RuntimeError, and
    `SimplificationRemoveUnusedVertices` removes such vertices. PCL's
    implementation also leaks the VTK filter on every call; it is
    replaced (compat/vtk_mesh_args.h).
    """

    cdef Subdivision* me

    def __cinit__(self, str filter="linear"):
        self.me = new Subdivision()
        self.set_FilterType(filter)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_FilterType(self, str filter):
        if filter not in _SUBDIVISION_FILTERS:
            raise ValueError("filter must be one of %s"
                             % ", ".join(_SUBDIVISION_FILTERS))
        self.me.setFilter(_SUBDIVISION_FILTERS[filter])

    def process(self, PointCloud cloud not None, polygons):
        result = _process(self.me, cloud, polygons)
        if result is None:
            used = set()
            for polygon in polygons:
                used.update(polygon)
            if len(used) < cloud.size:
                raise RuntimeError(
                    "VTK's subdivision failed: %d of the %d vertices are "
                    "used by no face, which the Loop filter cannot handle "
                    "(SimplificationRemoveUnusedVertices drops them)"
                    % (cloud.size - len(used), cloud.size))
            raise RuntimeError("VTK's subdivision returned no mesh")
        return result


cdef class MeshQuadricDecimationVTK:
    """Quadric-error decimation (pcl::MeshQuadricDecimationVTK, VTK's
    vtkQuadricDecimation): collapses edges, cheapest first, until
    `target_reduction_factor` of the triangles are gone (0.5, PCL's
    default, keeps about half).

        cloud, polygons = pcl.MeshQuadricDecimationVTK(0.9).process(
            cloud, polygons)

    The output's vertices are a new, smaller set; polygons index into it.
    """

    cdef QuadricDecimation* me

    def __cinit__(self, double target_reduction_factor=0.5):
        self.me = new QuadricDecimation()
        self.set_TargetReductionFactor(target_reduction_factor)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_TargetReductionFactor(self, double factor):
        if not 0 <= factor < 1:
            raise ValueError("target_reduction_factor must be in [0, 1)")
        self.me.setTargetReductionFactor(factor)

    def process(self, PointCloud cloud not None, polygons):
        result = _process(self.me, cloud, polygons)
        if result is None:
            raise RuntimeError("VTK's quadric decimation returned no mesh")
        return result


_READABLE = ("stl", "vtk", "ply", "obj")
_WRITABLE = ("stl", "vtk", "ply")


cdef str _extension(path, allowed):
    import os.path
    name = os.path.splitext(str(path))[1][1:].lower()
    if name not in allowed:
        raise ValueError("unsupported mesh format %r (%s)"
                         % (name, ", ".join(allowed)))
    return name


def load_polygon_file(path):
    """Read a mesh through VTK (pcl::io::loadPolygonFile): .stl, .vtk
    (legacy), .ply or .obj, by extension. Returns ``(cloud, polygons)``.
    STL stores triangles only, so its vertices come back merged and
    renumbered, without any no face used."""
    import os.path
    _extension(path, _READABLE)
    if not os.path.isfile(str(path)):
        raise IOError("no mesh file %s" % path)
    cdef string source = str(path).encode()
    cdef shared_ptr[cPointCloud[PointXYZ]] holder
    holder.reset(new cPointCloud[PointXYZ]())
    cdef cPointCloud[PointXYZ]* out = holder.get()
    cdef vector[int] indices, counts
    cdef int n
    with nogil:
        n = loadPolygonFileFlat(source, deref(out), indices, counts)
    if n <= 0:
        raise IOError("VTK read no mesh from %s" % path)
    return wrap_cloud(holder), _polygons(indices, counts)


def save_polygon_file(path, PointCloud cloud not None, polygons,
                      bint binary=True):
    """Write a mesh through VTK (pcl::io::savePolygonFile): .stl, .vtk
    (legacy) or .ply, by extension; `binary` picks the binary variant of
    each. STL holds triangles only: VTK triangulates other polygons and
    drops vertices no face uses."""
    _extension(path, _WRITABLE)
    cdef vector[int] indices, counts
    _flatten(polygons, cloud.size, indices, counts)
    cdef string target = str(path).encode()
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    cdef bint ok
    with nogil:
        ok = savePolygonFileFlat(target, deref(src), indices, counts, binary)
    if not ok:
        raise IOError("VTK could not write %s" % path)
