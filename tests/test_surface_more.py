"""Runtime tests for the second surface batch: OrganizedFastMesh,
GridProjection, BilateralUpsampling, SurfelSmoothing, EarClipping and
SimplificationRemoveUnusedVertices.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _pinhole(depth, f=40.0):
    """Back-project a (h, w) depth image through a pinhole camera with
    focal length f and the principal point at the image centre."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    x = (u - cx) * depth / f
    y = (v - cy) * depth / f
    return np.dstack([x, y, depth]).astype(np.float32)


def _organized(depth):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(_pinhole(depth))
    return cloud


def _flat(w=20, h=15, z=1.0):
    return _organized(np.full((h, w), z, dtype=np.float32))


# --- OrganizedFastMesh ---------------------------------------------------


def test_organized_fast_mesh_quads_by_default():
    cloud = _flat()
    faces = pcl.OrganizedFastMesh(cloud).reconstruct()
    assert len(faces) == 19 * 14
    assert all(len(face) == 4 for face in faces)
    assert max(max(face) for face in faces) < cloud.size


@pytest.mark.parametrize("kind", ["TRIANGLE_RIGHT_CUT", "TRIANGLE_LEFT_CUT",
                                  "TRIANGLE_ADAPTIVE_CUT"])
def test_organized_fast_mesh_triangles(kind):
    cloud = _flat()
    mesher = pcl.OrganizedFastMesh(cloud)
    mesher.set_TriangulationType(getattr(pcl, kind))
    faces = mesher.reconstruct()
    assert len(faces) == 2 * 19 * 14
    assert all(len(face) == 3 for face in faces)


def test_organized_fast_mesh_leaves_depth_jumps_open():
    depth = np.full((15, 20), 1.0, dtype=np.float32)
    depth[:, 10:] = 3.0
    mesher = pcl.OrganizedFastMesh(_organized(depth))
    mesher.set_TriangulationType(pcl.TRIANGLE_RIGHT_CUT)
    mesher.set_MaxEdgeLength(0.5)
    faces = mesher.reconstruct()
    # The column of quads straddling the jump is gone, both sides stay.
    assert 0 < len(faces) <= 2 * 18 * 14


def test_organized_fast_mesh_pixel_size_coarsens():
    """PCL steps by ``max(1, size - 1)`` pixels: 2 is still every pixel,
    3 skips one (20x15 grid -> 9 x 7 quads)."""
    mesher = pcl.OrganizedFastMesh(_flat())
    mesher.set_TrianglePixelSize(2)
    assert len(mesher.reconstruct()) == 19 * 14
    mesher.set_TrianglePixelSize(3)
    assert len(mesher.reconstruct()) == 9 * 7


def test_organized_fast_mesh_needs_an_organized_cloud():
    with pytest.raises(ValueError):
        pcl.OrganizedFastMesh(pcl.PointCloud(np.zeros((10, 3), np.float32)))
    with pytest.raises(RuntimeError):
        pcl.OrganizedFastMesh().reconstruct()
    with pytest.raises(ValueError):
        pcl.OrganizedFastMesh().set_TriangulationType(42)


def test_organized_fast_mesh_saves_as_a_mesh(tmp_path):
    cloud = _flat()
    faces = pcl.OrganizedFastMesh(cloud).reconstruct()
    path = str(tmp_path / "grid.ply")
    pcl.save_mesh(cloud, faces, path)
    _, loaded = pcl.load_mesh(path)
    assert len(loaded) == len(faces)


# --- GridProjection ------------------------------------------------------


def _sphere(n=400):
    rng = np.random.RandomState(0)
    normals = rng.randn(n, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = normals.astype(np.float32)
    joined = np.hstack([points, normals.astype(np.float32),
                        np.zeros((n, 1), np.float32)])
    return pcl.PointCloud_PointNormal(joined.astype(np.float32))


def test_grid_projection_meshes_a_sphere():
    mesher = pcl.GridProjection(_sphere())
    mesher.set_Resolution(0.2)
    mesher.set_PaddingSize(2)
    mesher.set_NearestNeighborNum(20)
    vertices, faces = mesher.reconstruct()

    assert isinstance(vertices, pcl.PointCloud_PointNormal)
    assert vertices.size > 0
    assert len(faces) > 0
    assert max(max(face) for face in faces) < vertices.size
    radius = np.linalg.norm(vertices.to_array()[:, :3], axis=1)
    assert np.median(np.abs(radius - 1.0)) < 0.2


def test_grid_projection_settings_round_trip():
    mesher = pcl.GridProjection()
    mesher.set_Resolution(0.05)
    assert mesher.get_Resolution() == pytest.approx(0.05)
    mesher.set_PaddingSize(4)
    assert mesher.get_PaddingSize() == 4
    mesher.set_NearestNeighborNum(12)
    assert mesher.get_NearestNeighborNum() == 12
    mesher.set_MaxBinarySearchLevel(7)
    assert mesher.get_MaxBinarySearchLevel() == 7
    with pytest.raises(ValueError):
        mesher.set_Resolution(0)
    with pytest.raises(RuntimeError):
        mesher.reconstruct()


# --- BilateralUpsampling -------------------------------------------------

F, CX, CY = 40.0, 20.0, 15.0
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]], dtype=np.float32)


def _rgbd_with_holes():
    h, w = 30, 40
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 2.0, dtype=np.float32)
    xyz = np.dstack([(u - CX) * z / F, (v - CY) * z / F, z]).astype(np.float32)
    rng = np.random.RandomState(4)
    holes = rng.rand(h, w) < 0.1
    xyz[holes] = np.nan
    rgba = np.full((h, w, 4), 128, dtype=np.uint8)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)
    return cloud, holes, u, v


def test_bilateral_upsampling_fills_holes_and_reprojects():
    cloud, holes, u, v = _rgbd_with_holes()
    assert cloud.is_organized
    upsampler = cloud.make_BilateralUpsampling()
    upsampler.set_ProjectionMatrix(K)
    upsampler.set_WindowSize(3)
    out = upsampler.process()

    assert isinstance(out, pcl.PointCloud_PointXYZRGBA)
    assert (out.width, out.height) == (40, 30)
    xyz = out.to_xyz_array().reshape(30, 40, 3)
    assert np.isfinite(xyz).all()
    # Every pixel, holes included, lands on the z=2 plane through K.
    assert xyz[..., 2] == pytest.approx(np.full((30, 40), 2.0), abs=1e-3)
    assert xyz[..., 0] == pytest.approx((u - CX) * 2.0 / F, abs=1e-3)
    assert xyz[..., 1] == pytest.approx((v - CY) * 2.0 / F, abs=1e-3)


def test_bilateral_upsampling_settings():
    upsampler = pcl.BilateralUpsampling()
    upsampler.set_SigmaColor(10.0)
    assert upsampler.get_SigmaColor() == pytest.approx(10.0)
    upsampler.set_SigmaDepth(1.5)
    assert upsampler.get_SigmaDepth() == pytest.approx(1.5)
    upsampler.set_WindowSize(4)
    assert upsampler.get_WindowSize() == 4
    with pytest.raises(ValueError):
        upsampler.set_ProjectionMatrix(np.eye(4))
    with pytest.raises(ValueError):
        upsampler.set_ProjectionMatrix(np.zeros((3, 3)))
    with pytest.raises(RuntimeError):
        upsampler.process()


def test_bilateral_upsampling_needs_an_organized_cloud():
    flat = pcl.PointCloud_PointXYZRGBA()
    flat.from_rgba_array(np.zeros((5, 3), np.float32),
                         np.zeros((5, 4), np.uint8))
    with pytest.raises(ValueError):
        pcl.BilateralUpsampling(flat)


# --- SurfelSmoothing -----------------------------------------------------


def test_surfel_smoothing_flattens_a_noisy_plane():
    rng = np.random.RandomState(2)
    points = np.zeros((600, 3), dtype=np.float32)
    points[:, :2] = rng.rand(600, 2)
    points[:, 2] = rng.randn(600) * 0.01
    cloud = pcl.PointCloud(points)
    normals = pcl.PointCloud_Normal(np.tile(
        np.array([0, 0, 1, 0], np.float32), (600, 1)))

    smoothed, smoothed_normals = pcl.SurfelSmoothing(scale=0.05).compute(
        cloud, normals)
    assert smoothed.size == 600
    assert smoothed_normals.size == 600
    assert np.std(smoothed.to_array()[:, 2]) < np.std(points[:, 2])


def test_surfel_smoothing_checks_its_inputs():
    with pytest.raises(ValueError):
        pcl.SurfelSmoothing(scale=0)
    smoother = pcl.SurfelSmoothing(scale=0.1)
    assert smoother.scale == pytest.approx(0.1)
    cloud = pcl.PointCloud(np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError):
        smoother.compute(cloud, pcl.PointCloud_Normal(
            np.zeros((3, 4), np.float32)))


# --- EarClipping / SimplificationRemoveUnusedVertices -------------------


def _area(points, triangle):
    a, b, c = (points[i] for i in triangle)
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


L_SHAPE = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],
                    [1, 2, 0], [0, 2, 0]], dtype=np.float32)


def test_ear_clipping_triangulates_convex_polygons():
    points = np.array([[0, 0, 0], [2, 0, 0], [3, 1.5, 0], [2, 3, 0],
                       [0, 3, 0], [-1, 1.5, 0], [5, 5, 0], [6, 5, 0],
                       [5, 6, 0]], dtype=np.float32)
    cloud = pcl.PointCloud(points)
    triangles = pcl.EarClipping().process(
        cloud, [(0, 1, 2, 3, 4, 5), (6, 7, 8)])
    assert all(len(t) == 3 for t in triangles)
    assert len(triangles) == 4 + 1
    hexagon = [t for t in triangles if max(t) <= 5]
    assert sum(_area(points, t) for t in hexagon) == pytest.approx(9.0)
    assert (6, 7, 8) in triangles


def test_ear_clipping_passes_concave_polygons_pcl_gets_right():
    """PCL clips this L-shape correctly from this vertex order; the
    result check must not reject a right answer."""
    triangles = pcl.EarClipping().process(pcl.PointCloud(L_SHAPE),
                                          [(5, 4, 3, 2, 1, 0)])
    assert len(triangles) == 4
    assert sum(_area(L_SHAPE, t) for t in triangles) == pytest.approx(3.0)


def test_ear_clipping_refuses_a_wrong_triangulation():
    """From this order PCL 1.14 covers the notch (area 4, not 3)."""
    with pytest.raises(RuntimeError, match="mis-triangulated polygon 0"):
        pcl.EarClipping().process(pcl.PointCloud(L_SHAPE),
                                  [(0, 1, 2, 3, 4, 5)])


def test_ear_clipping_rejects_bad_polygons():
    cloud = pcl.PointCloud(np.zeros((3, 3), np.float32))
    with pytest.raises(IndexError):
        pcl.EarClipping().process(cloud, [(0, 1, 7)])
    with pytest.raises(ValueError):
        pcl.EarClipping().process(cloud, [(0, 1)])


def test_remove_unused_vertices_compacts_the_mesh():
    rng = np.random.RandomState(5)
    points = rng.rand(10, 3).astype(np.float32)
    cloud = pcl.PointCloud(points)
    polygons = [(2, 5, 7), (5, 7, 9)]
    vertices, faces, kept = (
        pcl.SimplificationRemoveUnusedVertices().simplify(cloud, polygons))

    assert vertices.size == 4
    assert sorted(kept.tolist()) == [2, 5, 7, 9]
    assert vertices.to_array() == pytest.approx(points[kept])
    for old, new in zip(polygons, faces):
        assert vertices.to_array()[list(new)] == pytest.approx(
            points[list(old)])
