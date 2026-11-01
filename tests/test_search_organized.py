"""Runtime tests for OrganizedNeighbor (pcl::search::OrganizedNeighbor).

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

W, H, F = 40, 30, 50.0


def _camera_cloud():
    """A wavy (non-planar) surface seen through a pinhole camera."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = 1.0 + 0.3 * np.sin(u / 5.0) + 0.2 * np.cos(v / 4.0)
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    return cloud


def test_organized_neighbor_finds_each_point_itself():
    cloud = _camera_cloud()
    search = pcl.OrganizedNeighbor(cloud)
    indices, distances = search.nearest_k_search_for_cloud(cloud, k=1)
    assert indices.shape == (cloud.size, 1)
    assert np.mean(indices[:, 0] == np.arange(cloud.size)) > 0.99
    assert np.all(distances[:, 0] < 1e-6)


def test_organized_neighbor_agrees_with_kdtree():
    cloud = _camera_cloud()
    organized, _ = pcl.OrganizedNeighbor(cloud).nearest_k_search_for_cloud(
        cloud, k=4)
    kdtree, _ = pcl.KdTreeFLANN(cloud).nearest_k_search_for_cloud(cloud, k=4)
    agree = np.mean([set(a) == set(b) for a, b in zip(organized, kdtree)])
    assert agree > 0.9


def test_organized_neighbor_point_and_radius_search():
    cloud = _camera_cloud()
    search = pcl.OrganizedNeighbor(cloud)
    indices, _ = search.nearest_k_search_for_point(cloud, 100, k=3)
    assert indices[0] == 100
    found, distances = search.radius_search_for_cloud(cloud, 0.05,
                                                      max_nn=8)
    assert found.shape == (cloud.size, 8)
    assert np.all(distances <= 0.05 ** 2 + 1e-6)


def test_organized_neighbor_estimates_the_camera():
    matrix = pcl.OrganizedNeighbor(_camera_cloud()).get_CameraMatrix()
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(F, rel=0.05)
    assert matrix[1, 1] == pytest.approx(F, rel=0.05)
    assert matrix[0, 2] == pytest.approx(W / 2, abs=1.0)
    assert matrix[1, 2] == pytest.approx(H / 2, abs=1.0)


def test_organized_neighbor_refuses_non_camera_clouds():
    with pytest.raises(ValueError):
        pcl.OrganizedNeighbor(pcl.PointCloud(
            np.random.rand(50, 3).astype(np.float32)))
    scrambled = pcl.PointCloud()
    scrambled.from_organized_array(
        np.random.RandomState(0).rand(H, W, 3).astype(np.float32))
    with pytest.raises(ValueError):
        pcl.OrganizedNeighbor(scrambled)
    with pytest.raises(RuntimeError):
        pcl.OrganizedNeighbor().nearest_k_search_for_cloud(_camera_cloud())
