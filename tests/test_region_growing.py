"""Runtime tests for region growing segmentation: normals-driven
(`RegionGrowing`) and colour-driven (`RegionGrowingRGB`).

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (`make_RegionGrowing`,
`set_InputNormals`, `extract`).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

GRID = 20


@pytest.fixture
def two_faces():
    """An L shape: a floor grid at z=0 (the first 400 points) and a wall
    grid at x=1 rising above its far edge (the last 400). Normals are
    flat on either face and only disagree along the crease."""
    u, v = np.meshgrid(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID))
    u, v = u.ravel(), v.ravel()
    floor = np.column_stack([u, v, np.zeros_like(u)])
    wall = np.column_stack([np.ones_like(u), v, 0.05 + u])
    return pcl.PointCloud(np.vstack([floor, wall]).astype(np.float32))


def _grown(cloud, smoothness=np.deg2rad(5.0)):
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(10)
    grower = cloud.make_RegionGrowing()
    grower.set_InputNormals(normals.compute_cloud())
    grower.set_MinClusterSize(50)
    grower.set_NumberOfNeighbours(10)
    grower.set_SmoothnessThreshold(smoothness)
    grower.set_CurvatureThreshold(1.0)
    return grower


# --- RegionGrowing ----------------------------------------------------

def test_region_growing_separates_the_two_faces(two_faces):
    grower = _grown(two_faces)
    regions = grower.extract()
    assert len(regions) == 2
    sizes = sorted(len(r) for r in regions)
    # each face keeps almost all of its 400 points; only the crease is
    # ambiguous
    assert sizes[0] >= 380 and sizes[1] >= 380
    # and no region straddles the crease
    for region in regions:
        side = {i < 400 for i in region}
        assert len(side) == 1, "a region mixes floor and wall points"


def test_region_growing_is_reached_through_the_factory(two_faces):
    assert isinstance(two_faces.make_RegionGrowing(), pcl.RegionGrowing)


def test_segment_from_point_returns_that_points_region(two_faces):
    grower = _grown(two_faces)
    regions = grower.extract()
    floor_region = next(r for r in regions if 0 in r)
    assert sorted(grower.get_SegmentFromPoint(0)) == sorted(floor_region)
    wall_region = next(r for r in regions if 799 in r)
    assert sorted(grower.get_SegmentFromPoint(799)) == sorted(wall_region)


def test_segment_from_point_rejects_a_bad_index(two_faces):
    grower = _grown(two_faces)
    with pytest.raises(IndexError):
        grower.get_SegmentFromPoint(800)
    with pytest.raises(IndexError):
        grower.get_SegmentFromPoint(-1)


def test_colored_cloud_paints_each_region_one_colour(two_faces):
    grower = _grown(two_faces)
    regions = grower.extract()
    colored = grower.get_ColoredCloud()
    assert isinstance(colored, pcl.PointCloud_PointXYZRGB)
    assert colored.size == two_faces.size
    assert colored.to_xyz_array() == pytest.approx(
        two_faces.to_array()[:, :3], abs=1e-6)
    rgb = colored.to_rgb_array()
    colours = []
    for region in regions:
        region_colours = {tuple(rgb[i]) for i in region}
        assert len(region_colours) == 1, "a region is not one colour"
        colours.append(region_colours.pop())
    assert colours[0] != colours[1]


def test_colored_cloud_needs_extract_first(two_faces):
    grower = _grown(two_faces)
    with pytest.raises(RuntimeError, match="extract"):
        grower.get_ColoredCloud()


def test_region_growing_refuses_to_run_without_normals(two_faces):
    grower = two_faces.make_RegionGrowing()
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        grower.extract()
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        grower.get_SegmentFromPoint(0)


def test_a_loose_threshold_merges_the_faces(two_faces):
    """Allow 120 degrees between normals and the crease no longer stops
    the growth: everything is one region."""
    grower = _grown(two_faces, smoothness=np.deg2rad(120.0))
    regions = grower.extract()
    assert len(regions) == 1
    assert len(regions[0]) == two_faces.size


def test_region_growing_settings_roundtrip(two_faces):
    grower = two_faces.make_RegionGrowing()
    grower.set_MinClusterSize(7)
    grower.set_MaxClusterSize(7000)
    grower.set_NumberOfNeighbours(13)
    grower.set_SmoothnessThreshold(0.25)
    grower.set_CurvatureThreshold(0.75)
    grower.set_ResidualThreshold(0.125)
    grower.set_SmoothModeFlag(False)
    grower.set_CurvatureTestFlag(False)
    grower.set_ResidualTestFlag(True)
    assert grower.get_MinClusterSize() == 7
    assert grower.get_MaxClusterSize() == 7000
    assert grower.get_NumberOfNeighbours() == 13
    assert grower.get_SmoothnessThreshold() == pytest.approx(0.25)
    assert grower.get_CurvatureThreshold() == pytest.approx(0.75)
    assert grower.get_ResidualThreshold() == pytest.approx(0.125)
    assert grower.get_SmoothModeFlag() is False
    assert grower.get_CurvatureTestFlag() is False
    assert grower.get_ResidualTestFlag() is True


# --- RegionGrowingRGB -------------------------------------------------

@pytest.fixture
def two_colours():
    """A flat 30x30 grid, red on the left half and blue on the right."""
    n = 30
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    xyz = np.column_stack([u.ravel(), v.ravel(), np.zeros(n * n)])
    rgb = np.zeros((n * n, 3), dtype=np.uint8)
    left = u.ravel() < 0.5
    rgb[left] = (255, 0, 0)
    rgb[~left] = (0, 0, 255)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz.astype(np.float32), rgb)
    return cloud, left


def _grown_rgb(cloud):
    grower = cloud.make_RegionGrowingRGB()
    grower.set_DistanceThreshold(0.1)
    grower.set_PointColorThreshold(6.0)
    grower.set_RegionColorThreshold(5.0)
    grower.set_MinClusterSize(50)
    return grower


def test_region_growing_rgb_splits_by_colour(two_colours):
    cloud, left = two_colours
    regions = _grown_rgb(cloud).extract()
    assert len(regions) == 2
    for region in regions:
        side = {bool(left[i]) for i in region}
        assert len(side) == 1, "a region mixes red and blue points"
    assert sum(len(r) for r in regions) == cloud.size


def test_region_growing_rgb_is_reached_through_the_factory(two_colours):
    cloud, _ = two_colours
    assert isinstance(cloud.make_RegionGrowingRGB(), pcl.RegionGrowingRGB)


def test_region_growing_rgb_segment_from_point_and_colours(two_colours):
    cloud, left = two_colours
    grower = _grown_rgb(cloud)
    regions = grower.extract()
    first_red = int(np.flatnonzero(left)[0])
    region = grower.get_SegmentFromPoint(first_red)
    assert sorted(region) == sorted(next(r for r in regions
                                         if first_red in r))
    colored = grower.get_ColoredCloud()
    assert colored.size == cloud.size
    rgb = colored.to_rgb_array()
    assert len({tuple(rgb[i]) for i in regions[0]}) == 1
    assert tuple(rgb[regions[0][0]]) != tuple(rgb[regions[1][0]])


def test_region_growing_rgb_normal_test_needs_normals(two_colours):
    cloud, _ = two_colours
    grower = _grown_rgb(cloud)
    grower.set_NormalTestFlag(True)
    assert grower.get_NormalTestFlag() is True
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        grower.extract()


def test_region_growing_rgb_settings_roundtrip(two_colours):
    cloud, _ = two_colours
    grower = cloud.make_RegionGrowingRGB()
    grower.set_PointColorThreshold(7.5)
    grower.set_RegionColorThreshold(8.5)
    grower.set_DistanceThreshold(0.3)
    grower.set_NumberOfRegionNeighbours(11)
    grower.set_MinClusterSize(9)
    assert grower.get_PointColorThreshold() == pytest.approx(7.5)
    assert grower.get_RegionColorThreshold() == pytest.approx(8.5)
    assert grower.get_DistanceThreshold() == pytest.approx(0.3)
    assert grower.get_NumberOfRegionNeighbours() == 11
    assert grower.get_MinClusterSize() == 9
