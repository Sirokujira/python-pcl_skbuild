"""PCL's VTK mesh code: smoothing, subdivision, decimation and the
STL / VTK / PLY / OBJ files. Only built where PCL has VTK."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._vtk_mesh",
                    reason="this PCL was built without VTK")


def _grid(n=15, noise=0.0, seed=0):
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    z = np.random.RandomState(seed).randn(n * n) * noise
    pts = np.c_[xx.ravel(), yy.ravel(), z].astype(np.float32)
    tris = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            tris += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    return pcl.PointCloud(pts), tris


def _sphere(n=400, noise=0.0, seed=1):
    """A triangulated unit sphere (the convex hull of points on it)."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    hull = pcl.ConvexHull(pcl.PointCloud(pts.astype(np.float32)))
    cloud, polygons = hull.reconstruct_with_polygons()
    if noise:
        arr = cloud.to_array()
        arr *= 1 + noise * rng.randn(len(arr), 1).astype(np.float32)
        cloud = pcl.PointCloud(arr)
    return cloud, polygons


def _radii(cloud):
    return np.linalg.norm(cloud.to_array(), axis=1)


def _boundary(n):
    idx = np.arange(n * n).reshape(n, n)
    return np.unique(np.r_[idx[0], idx[-1], idx[:, 0], idx[:, -1]])


def test_laplacian_smooths_and_keeps_the_mesh():
    cloud, tris = _grid(noise=0.05)
    out, polygons = pcl.MeshSmoothingLaplacianVTK(num_iter=50,
                                                  relaxation_factor=0.2) \
        .process(cloud, tris)
    assert out.size == cloud.size and polygons == tris
    before = cloud.to_array()[:, 2].std()
    assert out.to_array()[:, 2].std() < 0.6 * before
    # more relaxation, smoother
    gentle, _ = pcl.MeshSmoothingLaplacianVTK(num_iter=50).process(cloud, tris)
    assert out.to_array()[:, 2].std() < gentle.to_array()[:, 2].std() < before
    # pinned boundary
    pinned, _ = pcl.MeshSmoothingLaplacianVTK(
        num_iter=50, relaxation_factor=0.2,
        boundary_smoothing=False).process(cloud, tris)
    edge = _boundary(15)
    assert np.allclose(pinned.to_array()[edge], cloud.to_array()[edge])
    assert not np.allclose(out.to_array()[edge], cloud.to_array()[edge])


def test_windowed_sinc_does_not_shrink():
    cloud, polygons = _sphere(noise=0.03)
    laplace, _ = pcl.MeshSmoothingLaplacianVTK(
        num_iter=200, relaxation_factor=0.2).process(cloud, polygons)
    sinc, out_polygons = pcl.MeshSmoothingWindowedSincVTK().process(
        cloud, polygons)
    assert len(out_polygons) == len(polygons)
    assert _radii(laplace).mean() < 0.6                 # shrunk
    assert _radii(sinc).mean() == pytest.approx(1.0, abs=0.02)
    assert _radii(sinc).std() < 0.65 * _radii(cloud).std()   # smoothed
    with pytest.raises(ValueError):
        pcl.MeshSmoothingWindowedSincVTK(pass_band=0)


@pytest.mark.parametrize("kind", ["linear", "loop", "butterfly"])
def test_subdivision_quadruples_the_triangles(kind):
    cloud, polygons = _sphere(200)
    out, out_polygons = pcl.MeshSubdivisionVTK(kind).process(cloud, polygons)
    assert len(out_polygons) == 4 * len(polygons)
    # one new vertex per edge (V - E + F = 2 for the closed hull)
    edges = len(polygons) * 3 // 2
    assert out.size == cloud.size + edges
    original = out.to_array()[:cloud.size]
    moved = np.abs(original - cloud.to_array()).max()
    if kind == "loop":
        assert moved > 1e-3                 # approximating
        assert _radii(out).max() <= 1 + 1e-5
    else:
        assert moved < 1e-6                 # interpolating
    if kind == "linear":                    # midpoints: inside the sphere
        assert _radii(out)[cloud.size:].max() < 1


def test_loop_subdivision_with_an_unused_vertex_raises():
    cloud, tris = _grid(5)
    arr = np.vstack([cloud.to_array(), [[5, 5, 5]]]).astype(np.float32)
    with pytest.raises(RuntimeError, match="used by no face"):
        pcl.MeshSubdivisionVTK("loop").process(pcl.PointCloud(arr), tris)
    out, _ = pcl.MeshSubdivisionVTK("linear").process(pcl.PointCloud(arr), tris)
    assert out.size > arr.shape[0]
    with pytest.raises(ValueError):
        pcl.MeshSubdivisionVTK("catmull_clark")


def test_quadric_decimation():
    cloud, polygons = _sphere(600)
    out, out_polygons = pcl.MeshQuadricDecimationVTK(0.75).process(cloud,
                                                                    polygons)
    assert 0.2 * len(polygons) <= len(out_polygons) <= 0.3 * len(polygons)
    assert out.size < cloud.size
    assert np.all([max(p) < out.size for p in out_polygons])
    assert np.allclose(_radii(out), 1, atol=0.1)
    with pytest.raises(ValueError):
        pcl.MeshQuadricDecimationVTK(1.0)


def test_input_checks_and_triangulation():
    cloud, _ = _grid(3)
    quad = [(0, 1, 4, 3)]
    out, polygons = pcl.MeshSmoothingLaplacianVTK(num_iter=0).process(cloud,
                                                                      quad)
    assert len(polygons) == 2 and all(len(p) == 3 for p in polygons)
    smoother = pcl.MeshSmoothingLaplacianVTK()
    with pytest.raises(ValueError, match="no polygons"):
        smoother.process(cloud, [])
    with pytest.raises(IndexError):
        smoother.process(cloud, [(0, 1, 99)])
    with pytest.raises(ValueError, match="3 vertices"):
        smoother.process(cloud, [(0, 1)])
    with pytest.raises(ValueError):
        smoother.set_RelaxationFactor(0)


def _same_triangles(back, back_polygons, cloud, polygons, atol=1e-3):
    """Equal faces up to vertex numbering (STL merges and renumbers) and
    text precision (the legacy VTK writer rounds)."""
    original = cloud.to_array()
    loaded = back.to_array()
    nearest = np.argmin(np.linalg.norm(loaded[:, None] - original[None],
                                       axis=2), axis=1)
    assert np.allclose(loaded, original[nearest], atol=atol)
    renamed = sorted(tuple(sorted(nearest[list(p)])) for p in back_polygons)
    return renamed == sorted(tuple(sorted(p)) for p in polygons)


@pytest.mark.parametrize("ext", ["stl", "vtk", "ply"])
@pytest.mark.parametrize("binary", [True, False])
def test_polygon_file_round_trip(tmp_path, ext, binary):
    cloud, polygons = _sphere(100)
    path = str(tmp_path / ("mesh." + ext))
    pcl.save_polygon_file(path, cloud, polygons, binary=binary)
    back, back_polygons = pcl.load_polygon_file(path)
    assert back.size == cloud.size
    assert _same_triangles(back, back_polygons, cloud, polygons)


def test_save_mesh_and_load_mesh_take_stl(tmp_path):
    cloud, polygons = _sphere(60)
    path = str(tmp_path / "mesh.stl")
    pcl.save_mesh(cloud, polygons, path)
    back, back_polygons = pcl.load_mesh(path)
    assert _same_triangles(back, back_polygons, cloud, polygons)
    # and the legacy VTK file pcl.save_mesh writes without VTK reads back
    vtk_path = str(tmp_path / "mesh.vtk")
    pcl.save_mesh(cloud, polygons, vtk_path)
    back, back_polygons = pcl.load_mesh(vtk_path)
    assert _same_triangles(back, back_polygons, cloud, polygons)
    obj_path = str(tmp_path / "mesh.obj")
    pcl.save_mesh(cloud, polygons, obj_path)
    back, back_polygons = pcl.load_polygon_file(obj_path)
    assert _same_triangles(back, back_polygons, cloud, polygons)


def test_polygon_file_errors(tmp_path):
    cloud, polygons = _sphere(30)
    with pytest.raises(IOError):
        pcl.load_polygon_file(str(tmp_path / "missing.stl"))
    with pytest.raises(ValueError):
        pcl.save_polygon_file(str(tmp_path / "mesh.xyz"), cloud, polygons)
    with pytest.raises(ValueError):
        pcl.load_polygon_file(str(tmp_path / "mesh.ifs"))
    junk = tmp_path / "junk.stl"
    junk.write_bytes(b"not a mesh")
    with pytest.raises(IOError):
        pcl.load_polygon_file(str(junk))
