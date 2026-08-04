"""Runtime tests for the octree wrappers.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (pcl/pxi/Octree/).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

RESOLUTION = 0.1


@pytest.fixture
def cloud():
    points = np.random.RandomState(1).rand(1000, 3).astype(np.float32)
    return pcl.PointCloud(points)


def test_octree_is_populated_from_the_cloud(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    assert octree.get_resolution() == pytest.approx(RESOLUTION)
    assert octree.get_tree_depth() > 0
    # Fewer voxels than points, because points share voxels.
    centers = octree.get_occupied_voxel_centers()
    assert 0 < len(centers) < cloud.size
    assert all(len(c) == 3 for c in centers)


def test_octree_rejects_a_nonpositive_resolution():
    with pytest.raises(ValueError):
        pcl.OctreePointCloudSearch(0.0)


def test_voxel_centers_lie_inside_the_cloud_bounds(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    centers = np.array(octree.get_occupied_voxel_centers(), dtype=np.float32)
    points = cloud.to_array()
    assert (centers.min(axis=0) >= points.min(axis=0) - RESOLUTION).all()
    assert (centers.max(axis=0) <= points.max(axis=0) + RESOLUTION).all()


def test_nearest_k_search_finds_each_point_itself(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    ind, sqdist = octree.nearest_k_search_for_cloud(cloud, 3)
    assert ind.shape == (cloud.size, 3)
    assert (ind[:, 0] == np.arange(cloud.size)).all()
    assert sqdist[:, 0] == pytest.approx(0.0, abs=1e-6)


def test_nearest_k_search_rejects_bad_k(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    with pytest.raises(ValueError):
        octree.nearest_k_search_for_cloud(cloud, 0)


def test_radius_search_returns_only_neighbours_in_range(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    radius = 0.2
    indices, sqr_distances = octree.radius_search(cloud, 0, radius)
    assert len(indices) == len(sqr_distances)
    assert indices, "expected at least the query point itself"
    assert max(sqr_distances) <= radius ** 2 + 1e-6


def test_radius_search_honours_max_nn(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    assert len(octree.radius_search(cloud, 0, 0.5, max_nn=4)[0]) <= 4


def test_radius_search_index_out_of_range(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    with pytest.raises(IndexError):
        octree.radius_search(cloud, cloud.size, 0.1)


def test_voxel_search_groups_points_sharing_a_voxel():
    """Two points inside one voxel see each other; a distant third does
    not. (PCL's voxelSearch occasionally misses a query point on a voxel
    boundary — reproduced in plain C++ — so this uses points placed well
    inside their voxels rather than asserting over a random cloud.)"""
    points = np.array([[0.05, 0.05, 0.05],
                       [0.06, 0.05, 0.05],
                       [0.95, 0.95, 0.95]], dtype=np.float32)
    cloud = pcl.PointCloud(points)
    octree = cloud.make_octreeSearch(RESOLUTION)
    assert sorted(octree.voxel_search(cloud, 0)) == [0, 1]
    assert octree.voxel_search(cloud, 2) == [2]


def test_define_bounding_box_argument_count(cloud):
    octree = pcl.OctreePointCloudSearch(RESOLUTION)
    octree.set_InputCloud(cloud)
    octree.define_bounding_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    octree.add_points_from_input_cloud()
    assert octree.get_occupied_voxel_centers()
    with pytest.raises(TypeError):
        octree.define_bounding_box(1.0, 2.0)


def test_delete_tree_empties_it(cloud):
    octree = cloud.make_octreeSearch(RESOLUTION)
    assert octree.get_occupied_voxel_centers()
    octree.delete_tree()
    assert octree.get_occupied_voxel_centers() == []


# --- change detection ------------------------------------------------

def test_change_detector_finds_only_the_new_region():
    rng = np.random.RandomState(2)
    before = rng.rand(500, 3).astype(np.float32)
    added = (rng.rand(200, 3) + 5.0).astype(np.float32)

    detector = pcl.PointCloud(before).make_octreeChangeDetector(RESOLUTION)
    detector.switch_buffers()
    after = pcl.PointCloud(np.vstack([before, added]))
    detector.set_InputCloud(after)
    detector.add_points_from_input_cloud()

    new_indices = set(detector.get_PointIndicesFromNewVoxels())
    assert new_indices
    # Everything reported as new comes from the appended block -- except
    # index 0, which PCL always reports (see the unchanged-cloud test).
    assert all(i >= len(before) for i in new_indices - {0})
    assert len(new_indices) >= len(added) // 2


def test_change_detector_reports_nothing_new_for_an_unchanged_cloud():
    """Re-adding the same cloud finds no new region.

    PCL reports index 0 anyway -- its octree treats the first point
    specially, the same quirk that makes voxelSearch miss it (both
    reproduced in plain C++ against PCL 1.14.0, so neither is a binding
    bug). Everything beyond that single index must be absent.
    """
    points = np.random.RandomState(3).rand(400, 3).astype(np.float32)
    cloud = pcl.PointCloud(points)
    detector = cloud.make_octreeChangeDetector(RESOLUTION)
    detector.switch_buffers()
    detector.set_InputCloud(cloud)
    detector.add_points_from_input_cloud()
    assert set(detector.get_PointIndicesFromNewVoxels()) <= {0}


def test_change_detector_rejects_a_nonpositive_resolution():
    with pytest.raises(ValueError):
        pcl.OctreePointCloudChangeDetector(-1.0)
