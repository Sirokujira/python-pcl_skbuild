"""Runtime tests for triangulation, hull polygons and CropHull.

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
    """1500 points on the unit sphere — a closed surface to triangulate."""
    rng = np.random.RandomState(0)
    theta = rng.rand(1500) * 2 * np.pi
    phi = np.arccos(2 * rng.rand(1500) - 1)
    return pcl.PointCloud(np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ]).astype(np.float32))


@pytest.fixture
def sphere_normals(sphere):
    """Analytic normals: on a unit sphere they are the points themselves."""
    values = np.zeros((sphere.size, 4), dtype=np.float32)
    values[:, :3] = sphere.to_array()
    return pcl.PointCloud_Normal(values)


@pytest.fixture
def cube_region():
    """A cube of points occupying part of the unit cube."""
    rng = np.random.RandomState(1)
    return pcl.PointCloud(
        (rng.rand(200, 3) * 0.4 + 0.3).astype(np.float32))


# --- triangulation -----------------------------------------------------

def test_triangulation_returns_index_triples(sphere, sphere_normals):
    mesher = sphere.make_GreedyProjectionTriangulation()
    mesher.set_InputCloud(sphere, sphere_normals)
    mesher.set_SearchRadius(0.3)
    mesher.set_Mu(2.5)
    mesher.set_MaximumNearestNeighbors(100)
    triangles = mesher.reconstruct()

    assert triangles
    assert all(len(t) == 3 for t in triangles)
    # Vertices index the original cloud, so the points never cross back.
    assert max(max(t) for t in triangles) < sphere.size
    assert min(min(t) for t in triangles) >= 0


def test_triangulation_covers_most_of_the_surface(sphere, sphere_normals):
    mesher = sphere.make_GreedyProjectionTriangulation()
    mesher.set_InputCloud(sphere, sphere_normals)
    mesher.set_SearchRadius(0.3)
    mesher.set_Mu(2.5)
    mesher.set_MaximumNearestNeighbors(100)
    triangles = mesher.reconstruct()

    used = {index for triangle in triangles for index in triangle}
    assert len(used) > sphere.size * 0.5


def test_triangulation_needs_matching_lengths(sphere):
    mesher = pcl.GreedyProjectionTriangulation()
    short = pcl.PointCloud_Normal(np.zeros((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="same length"):
        mesher.set_InputCloud(sphere, short)


def test_triangulation_without_input_reports_it():
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.GreedyProjectionTriangulation().reconstruct()


def test_triangulation_settings_roundtrip(sphere, sphere_normals):
    mesher = sphere.make_GreedyProjectionTriangulation()
    mesher.set_InputCloud(sphere, sphere_normals)
    mesher.set_SearchRadius(0.25)
    mesher.set_Mu(3.0)
    mesher.set_MaximumNearestNeighbors(50)
    assert mesher.get_SearchRadius() == pytest.approx(0.25)
    assert mesher.get_Mu() == pytest.approx(3.0)
    assert mesher.get_MaximumNearestNeighbors() == 50


# --- hull polygons -----------------------------------------------------

def test_convex_hull_returns_points_and_polygons(cube_region):
    hull = cube_region.make_ConvexHull()
    hull.set_Dimension(3)
    points, polygons = hull.reconstruct_with_polygons()

    assert isinstance(points, pcl.PointCloud)
    assert 0 < points.size < cube_region.size
    assert polygons
    assert all(len(p) >= 3 for p in polygons)
    assert max(max(p) for p in polygons) < points.size


def test_concave_hull_returns_points_and_polygons(cube_region):
    hull = cube_region.make_ConcaveHull()
    hull.set_Dimension(3)
    hull.set_Alpha(0.5)
    points, polygons = hull.reconstruct_with_polygons()
    assert points.size > 0
    assert max(max(p) for p in polygons) < points.size


def test_polygon_reconstruct_agrees_with_the_plain_one(cube_region):
    hull = cube_region.make_ConvexHull()
    hull.set_Dimension(3)
    plain = hull.reconstruct().size
    with_polygons, _ = hull.reconstruct_with_polygons()
    assert with_polygons.size == plain


# --- crop hull ---------------------------------------------------------

def test_crop_hull_keeps_only_what_is_inside(cube_region):
    hull = cube_region.make_ConvexHull()
    hull.set_Dimension(3)
    hull_points, polygons = hull.reconstruct_with_polygons()

    scene = pcl.PointCloud(
        np.random.RandomState(2).rand(1000, 3).astype(np.float32))
    crop = scene.make_crophull()
    crop.set_HullCloud(hull_points)
    crop.set_HullIndices(polygons)
    crop.set_Dim(3)
    inside = crop.filter()

    assert 0 < inside.size < scene.size
    kept = inside.to_array()
    # The hull was built from points in [0.3, 0.7]; allow the hull's own
    # tolerance either side.
    assert (kept.min(axis=0) >= 0.28).all()
    assert (kept.max(axis=0) <= 0.72).all()


def test_crop_hull_outside_is_the_complement(cube_region):
    hull = cube_region.make_ConvexHull()
    hull.set_Dimension(3)
    hull_points, polygons = hull.reconstruct_with_polygons()
    scene = pcl.PointCloud(
        np.random.RandomState(3).rand(1000, 3).astype(np.float32))

    sizes = []
    for crop_outside in (True, False):
        crop = scene.make_crophull()
        crop.set_HullCloud(hull_points)
        crop.set_HullIndices(polygons)
        crop.set_Dim(3)
        crop.set_CropOutside(crop_outside)
        sizes.append(crop.filter().size)
    assert sizes[0] != sizes[1]
    assert sum(sizes) == scene.size


def test_crop_hull_keeps_its_hull_cloud_alive(cube_region):
    """PCL holds the hull by shared_ptr; the Python object owning that
    handle has to outlive the filter call too."""
    hull = cube_region.make_ConvexHull()
    hull.set_Dimension(3)
    hull_points, polygons = hull.reconstruct_with_polygons()

    scene = pcl.PointCloud(
        np.random.RandomState(4).rand(500, 3).astype(np.float32))
    crop = scene.make_crophull()
    crop.set_HullCloud(hull_points)
    crop.set_HullIndices(polygons)
    crop.set_Dim(3)
    del hull_points
    assert crop.filter().size >= 0
