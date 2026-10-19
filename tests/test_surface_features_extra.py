"""Runtime tests for Poisson reconstruction and the boundary / ESF / CVFH
features.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def sphere():
    """2000 points on the unit sphere."""
    rng = np.random.RandomState(0)
    theta = rng.rand(2000) * 2 * np.pi
    phi = np.arccos(2 * rng.rand(2000) - 1)
    return pcl.PointCloud(np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ]).astype(np.float32))


@pytest.fixture
def sphere_normals(sphere):
    """Analytic outward normals: on a unit sphere, the points themselves."""
    values = np.zeros((sphere.size, 4), dtype=np.float32)
    values[:, :3] = sphere.to_array()
    return pcl.PointCloud_Normal(values)


@pytest.fixture
def grid():
    """A flat 30x30 grid on the unit square."""
    u, v = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    return pcl.PointCloud(np.column_stack([
        u.ravel(), v.ravel(), np.zeros(900)]).astype(np.float32))


def _estimated_normals(cloud, k=20):
    estimator = cloud.make_NormalEstimation()
    estimator.set_KSearch(k)
    return estimator.compute_cloud()


# --- Poisson ------------------------------------------------------------

def test_poisson_rebuilds_the_sphere(sphere, sphere_normals):
    poisson = sphere.make_Poisson()
    poisson.set_InputCloud(sphere, sphere_normals)
    poisson.set_Depth(6)
    vertices, triangles = poisson.reconstruct()

    assert isinstance(vertices, pcl.PointCloud_PointNormal)
    assert vertices.size > 100
    assert len(triangles) > 100
    # the mesh has its OWN vertices, all close to the unit sphere
    radii = np.linalg.norm(vertices.to_array()[:, :3], axis=1)
    assert np.abs(radii - 1.0).mean() < 0.1
    # triangles, indexing into those vertices
    assert all(len(t) == 3 for t in triangles)
    flat = np.concatenate([np.asarray(t) for t in triangles])
    assert flat.min() >= 0 and flat.max() < vertices.size


def test_poisson_takes_a_pointnormal_cloud_directly(sphere, sphere_normals):
    joined = pcl.PointCloud_PointNormal().from_xyz_and_normals(
        sphere, sphere_normals)
    poisson = joined.make_Poisson()
    assert isinstance(poisson, pcl.Poisson)
    poisson.set_Depth(5)
    vertices, triangles = poisson.reconstruct()
    assert vertices.size > 50 and len(triangles) > 50


def test_poisson_mesh_can_be_saved_and_loaded(tmp_path, sphere, sphere_normals):
    poisson = sphere.make_Poisson()
    poisson.set_InputCloud(sphere, sphere_normals)
    poisson.set_Depth(5)
    vertices, triangles = poisson.reconstruct()
    path = tmp_path / "sphere.ply"
    pcl.save_mesh(vertices.to_xyz_cloud(), triangles, str(path))
    cloud, polygons = pcl.load_mesh(str(path))
    assert cloud.size == vertices.size
    assert len(polygons) == len(triangles)


def test_poisson_can_hand_back_polygons(sphere, sphere_normals):
    poisson = sphere.make_Poisson()
    poisson.set_InputCloud(sphere, sphere_normals)
    poisson.set_Depth(5)
    poisson.set_OutputPolygons(True)
    assert poisson.get_OutputPolygons() is True
    _, polygons = poisson.reconstruct()
    assert len(polygons) > 50
    assert all(len(p) >= 3 for p in polygons)


def test_poisson_input_is_checked(sphere, sphere_normals):
    poisson = sphere.make_Poisson()
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        poisson.reconstruct()
    with pytest.raises(TypeError, match="normals"):
        poisson.set_InputCloud(sphere)
    with pytest.raises(ValueError, match="points"):
        poisson.set_InputCloud(sphere, pcl.PointCloud_Normal(3))
    with pytest.raises(ValueError, match="empty"):
        poisson.set_InputCloud(pcl.PointCloud_PointNormal())
    with pytest.raises(TypeError):
        poisson.set_InputCloud("not a cloud")


def test_poisson_settings_roundtrip(sphere):
    poisson = sphere.make_Poisson()
    poisson.set_Depth(7)
    poisson.set_MinDepth(3)
    poisson.set_PointWeight(2.5)
    poisson.set_Scale(1.25)
    poisson.set_SolverDivide(6)
    poisson.set_IsoDivide(6)
    poisson.set_SamplesPerNode(4.0)
    poisson.set_Confidence(True)
    poisson.set_Degree(3)
    poisson.set_Manifold(False)
    assert poisson.get_Depth() == 7
    assert poisson.get_MinDepth() == 3
    assert poisson.get_PointWeight() == pytest.approx(2.5)
    assert poisson.get_Scale() == pytest.approx(1.25)
    assert poisson.get_SolverDivide() == 6
    assert poisson.get_IsoDivide() == 6
    assert poisson.get_SamplesPerNode() == pytest.approx(4.0)
    assert poisson.get_Confidence() is True
    assert poisson.get_Degree() == 3
    assert poisson.get_Manifold() is False


# --- BoundaryEstimation -------------------------------------------------

def test_boundary_flags_the_rim_of_a_grid(grid):
    edges = grid.make_BoundaryEstimation()
    edges.set_InputNormals(_estimated_normals(grid))
    edges.set_KSearch(20)
    on_edge = edges.compute()

    assert on_edge.dtype == np.bool_
    assert on_edge.shape == (grid.size,)
    xy = grid.to_array()[:, :2]
    rim = (xy.min(axis=1) < 1e-6) | (xy.max(axis=1) > 1 - 1e-6)
    interior = (xy.min(axis=1) > 0.2) & (xy.max(axis=1) < 0.8)
    assert on_edge[rim].all()
    assert not on_edge[interior].any()


def test_boundary_needs_normals(grid):
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        grid.make_BoundaryEstimation().compute()


def test_boundary_settings_roundtrip(grid):
    edges = grid.make_BoundaryEstimation()
    edges.set_AngleThreshold(1.0)
    edges.set_KSearch(12)
    assert edges.get_AngleThreshold() == pytest.approx(1.0)
    assert edges.get_KSearch() == 12


# --- ESF ----------------------------------------------------------------

def test_esf_returns_a_640_bin_descriptor(sphere):
    histogram = sphere.make_ESFEstimation().compute()
    assert histogram.shape == (640,)
    assert histogram.dtype == np.float32
    assert np.isfinite(histogram).all()
    assert (histogram >= 0).all()
    assert histogram.sum() > 0


def test_esf_distinguishes_a_sphere_from_a_plane(sphere, grid):
    ball = sphere.make_ESFEstimation().compute()
    flat = grid.make_ESFEstimation().compute()
    assert not np.allclose(ball, flat, atol=1e-3)


def test_esf_rejects_an_empty_cloud():
    with pytest.raises(RuntimeError, match="empty"):
        pcl.PointCloud().make_ESFEstimation().compute()


# --- CVFH ---------------------------------------------------------------

def test_cvfh_gives_one_descriptor_per_smooth_region(sphere, sphere_normals, grid):
    """A flat grid is one smooth region; an L shape (floor + wall) is two.

    Both need `set_RadiusNormals` raised above the grid spacing (0.05):
    CVFH re-estimates normals within that radius before clustering, and
    at PCL's default 0.03 every point is alone, no region forms, and the
    whole-cloud fallback hides the clustering entirely.
    """
    cvfh = grid.make_CVFHEstimation()
    cvfh.set_InputNormals(_estimated_normals(grid))
    cvfh.set_ClusterTolerance(0.1)
    cvfh.set_RadiusNormals(0.15)
    cvfh.set_MinPoints(50)
    flat = cvfh.compute()
    assert flat.shape == (1, 308)
    assert flat.dtype == np.float32
    assert flat.sum() > 0

    u, v = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    u, v = u.ravel(), v.ravel()
    floor = np.column_stack([u, v, np.zeros_like(u)])
    wall = np.column_stack([np.ones_like(u), v, 0.05 + u])
    l_shape = pcl.PointCloud(np.vstack([floor, wall]).astype(np.float32))
    cvfh = l_shape.make_CVFHEstimation()
    cvfh.set_InputNormals(_estimated_normals(l_shape, k=10))
    cvfh.set_ClusterTolerance(0.1)
    cvfh.set_RadiusNormals(0.15)
    cvfh.set_MinPoints(50)
    assert cvfh.compute().shape[0] >= 2


def test_cvfh_falls_back_to_one_descriptor_when_nothing_clusters(sphere, sphere_normals):
    """PCL's default cluster tolerance (0.015) is far below the spacing
    of this sphere, so no region forms and a single VFH comes back."""
    cvfh = sphere.make_CVFHEstimation()
    cvfh.set_InputNormals(sphere_normals)
    assert cvfh.compute().shape == (1, 308)


def test_cvfh_needs_normals(sphere):
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        sphere.make_CVFHEstimation().compute()
