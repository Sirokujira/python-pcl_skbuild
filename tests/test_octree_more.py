"""Runtime tests for octree ray casting, OctreePointCloudVoxelCentroid
and OctreePointCloudDensity.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def rod():
    """Points along the x axis from 0 to 1 plus a scattered cluster far
    off it."""
    rng = np.random.RandomState(0)
    line = np.zeros((200, 3), dtype=np.float32)
    line[:, 0] = np.linspace(0.0, 1.0, 200)
    far = (rng.rand(100, 3) * 0.2 + [0.4, 2.0, 2.0]).astype(np.float32)
    return pcl.PointCloud(np.vstack([line, far]))


# The rod's first point is the origin, so PCL puts a voxel corner there
# and the rod runs along voxel edges; the rays below go through the
# middle of the voxel row it occupies instead.
ORIGIN = (-1.0, 0.05, 0.05)
ALONG_X = (1.0, 0.0, 0.0)


def test_ray_hits_the_voxels_along_it_nearest_first(rod):
    octree = rod.make_octreeSearch(0.1)
    centers = octree.get_intersected_voxel_centers(ORIGIN, ALONG_X)
    assert centers.shape[1] == 3
    assert len(centers) >= 8
    assert np.all(np.diff(centers[:, 0]) > 0)
    assert centers[:, 1] == pytest.approx(np.full(len(centers), 0.05))


def test_ray_indices_are_the_points_inside(rod):
    octree = rod.make_octreeSearch(0.1)
    indices = octree.get_intersected_voxel_indices(ORIGIN, ALONG_X)
    assert indices.dtype == np.int32
    assert len(indices) > 150
    assert np.all(indices < 200)


def test_ray_max_voxel_count_is_exact(rod):
    """PCL applies the limit per subtree and overshoots; the wrapper
    returns exactly the nearest `max_voxel_count` voxels."""
    octree = rod.make_octreeSearch(0.1)
    every = octree.get_intersected_voxel_centers(ORIGIN, ALONG_X)
    for limit in (1, 2, 3, 5):
        first = octree.get_intersected_voxel_centers(
            ORIGIN, ALONG_X, max_voxel_count=limit)
        assert first == pytest.approx(every[:limit])
        indices = octree.get_intersected_voxel_indices(
            ORIGIN, ALONG_X, max_voxel_count=limit)
        points = rod.to_array()[indices]
        assert len(points) > 0
        assert points[:, 0].max() < first[-1, 0] + 0.05 + 1e-6


def test_ray_misses(rod):
    octree = rod.make_octreeSearch(0.1)
    miss = octree.get_intersected_voxel_centers((-1.0, -5.0, -5.0),
                                                (1.0, 0.0, 0.0))
    assert miss.shape == (0, 3)
    with pytest.raises(ValueError):
        octree.get_intersected_voxel_centers((0, 0, 0), (0, 0, 0))


@pytest.fixture
def two_clusters():
    """A lone first point at the origin -- where PCL then puts a voxel
    corner -- and two clusters in the middle of the 0.25 voxels
    [0.25, 0.5) and [0.75, 1.0)."""
    rng = np.random.RandomState(1)
    a = (rng.rand(30, 3) * 0.02 + 0.365).astype(np.float32)
    b = (rng.rand(50, 3) * 0.02 + 0.865).astype(np.float32)
    first = np.zeros((1, 3), dtype=np.float32)
    return pcl.PointCloud(np.vstack([first, a, b])), a, b


def test_voxel_centroids_are_the_cluster_means(two_clusters):
    cloud, a, b = two_clusters
    octree = cloud.make_octreeVoxelCentroid(0.25)
    assert octree.get_leaf_count() == 3
    centroids = octree.get_voxel_centroids()
    assert centroids.shape == (3, 3)
    order = np.argsort(centroids[:, 0])
    assert centroids[order[0]] == pytest.approx([0, 0, 0], abs=1e-6)
    assert centroids[order[1]] == pytest.approx(a.mean(axis=0), abs=1e-5)
    assert centroids[order[2]] == pytest.approx(b.mean(axis=0), abs=1e-5)


def test_voxel_centroid_at_point(two_clusters):
    cloud, a, _ = two_clusters
    octree = cloud.make_octreeVoxelCentroid(0.25)
    found = octree.get_voxel_centroid_at_point(tuple(a[0]))
    assert found == pytest.approx(a.mean(axis=0), abs=1e-5)
    assert octree.get_voxel_centroid_at_point((0.625, 0.625, 0.625)) is None
    assert octree.get_occupied_voxel_centers().shape == (3, 3)
    assert octree.is_voxel_occupied_at_point(tuple(a[0]))


def test_voxel_density_counts_points(two_clusters):
    cloud, a, b = two_clusters
    octree = cloud.make_octreeDensity(0.25)
    assert octree.get_voxel_density_at_point(tuple(a[0])) == 30
    assert octree.get_voxel_density_at_point(tuple(b[0])) == 50
    assert octree.get_voxel_density_at_point((0.625, 0.625, 0.625)) == 0
    assert octree.get_voxel_density_at_point((0.0, 0.0, 0.0)) == 1
    assert octree.get_leaf_count() == 3
    assert octree.get_resolution() == pytest.approx(0.25)


def test_bounding_box_sets_the_grid():
    """Without a bounding box the first point is a voxel corner and a
    cluster around it splits; with one, the grid starts where asked."""
    rng = np.random.RandomState(2)
    cluster = pcl.PointCloud(
        (rng.rand(40, 3) * 0.02 + 0.115).astype(np.float32))
    split = cluster.make_octreeDensity(0.25)
    assert split.get_leaf_count() > 1

    octree = pcl.OctreePointCloudDensity(0.25)
    octree.define_bounding_box((0, 0, 0), (1, 1, 1))
    octree.set_InputCloud(cluster)
    octree.add_points_from_input_cloud()
    assert octree.get_leaf_count() == 1
    assert octree.get_voxel_density_at_point((0.125, 0.125, 0.125)) == 40
    with pytest.raises(RuntimeError):
        octree.define_bounding_box((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError):
        pcl.OctreePointCloudVoxelCentroid(0.25).define_bounding_box(
            (0, 0, 0), (0, 1, 1))


def test_octree_resolution_must_be_positive():
    with pytest.raises(ValueError):
        pcl.OctreePointCloudVoxelCentroid(0)
    with pytest.raises(ValueError):
        pcl.OctreePointCloudDensity(-1)
