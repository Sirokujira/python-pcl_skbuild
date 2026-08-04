"""Runtime tests for organized clouds, VFH, bilateral filtering and
min-cut segmentation.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

GRID = 40


@pytest.fixture
def noisy_depth_grid():
    """A 40x40 depth image at z=1 with a little noise — the organized
    layout FastBilateralFilter needs."""
    rng = np.random.RandomState(0)
    grid = np.zeros((GRID, GRID, 3), dtype=np.float32)
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    grid[:, :, 0] = cols * 0.05
    grid[:, :, 1] = rows * 0.05
    grid[:, :, 2] = 1.0 + rng.normal(0.0, 0.02, (GRID, GRID))
    return grid


# --- organized clouds --------------------------------------------------

def test_organized_cloud_keeps_its_grid(noisy_depth_grid):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(noisy_depth_grid)
    assert cloud.is_organized
    assert cloud.width == GRID
    assert cloud.height == GRID
    assert cloud.size == GRID * GRID


def test_organized_roundtrip(noisy_depth_grid):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(noisy_depth_grid)
    assert cloud.to_organized_array() == pytest.approx(
        noisy_depth_grid, abs=1e-6)


def test_unorganized_cloud_says_so():
    cloud = pcl.PointCloud(np.zeros((10, 3), dtype=np.float32))
    assert not cloud.is_organized
    with pytest.raises(ValueError, match="not organized"):
        cloud.to_organized_array()


def test_organized_array_shape_is_checked():
    cloud = pcl.PointCloud()
    with pytest.raises(ValueError, match="height, width, 3"):
        cloud.from_organized_array(np.zeros((4, 4, 2), dtype=np.float32))


# --- fast bilateral ----------------------------------------------------

def test_fast_bilateral_smooths_the_depth_image(noisy_depth_grid):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(noisy_depth_grid)

    smoother = cloud.make_FastBilateralFilter()
    smoother.set_sigma_s(5.0)
    smoother.set_sigma_r(0.05)
    smoothed = smoother.filter()

    assert smoothed.size == cloud.size
    assert smoothed.is_organized
    before = np.abs(noisy_depth_grid[:, :, 2] - 1.0).mean()
    after = np.abs(smoothed.to_organized_array()[:, :, 2] - 1.0).mean()
    assert after < before


def test_fast_bilateral_settings_roundtrip(noisy_depth_grid):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(noisy_depth_grid)
    smoother = cloud.make_FastBilateralFilter()
    smoother.set_sigma_s(7.5)
    smoother.set_sigma_r(0.03)
    assert smoother.get_sigma_s() == pytest.approx(7.5)
    assert smoother.get_sigma_r() == pytest.approx(0.03)


# --- VFH ---------------------------------------------------------------

def test_vfh_returns_a_308_bin_descriptor():
    cloud = pcl.PointCloud(
        np.random.RandomState(1).rand(500, 3).astype(np.float32))
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(20)

    vfh = cloud.make_VFHEstimation()
    vfh.set_InputNormals(normals.compute_cloud())
    descriptor = vfh.compute()

    assert descriptor.shape == (308,)
    assert descriptor.dtype == np.float32
    assert descriptor.sum() > 0


def test_vfh_distinguishes_two_shapes():
    """The point of a descriptor: different geometry, different vector."""
    rng = np.random.RandomState(2)
    flat = np.zeros((500, 3), dtype=np.float32)
    flat[:, :2] = rng.rand(500, 2)
    blob = rng.rand(500, 3).astype(np.float32)

    descriptors = []
    for points in (flat, blob):
        cloud = pcl.PointCloud(points)
        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(20)
        vfh = cloud.make_VFHEstimation()
        vfh.set_InputNormals(normals.compute_cloud())
        descriptors.append(vfh.compute())

    assert not np.allclose(descriptors[0], descriptors[1])


def test_vfh_without_normals_reports_the_reason():
    cloud = pcl.PointCloud(
        np.random.RandomState(3).rand(100, 3).astype(np.float32))
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        cloud.make_VFHEstimation().compute()


# --- min cut -----------------------------------------------------------

def test_min_cut_separates_foreground_from_background():
    rng = np.random.RandomState(4)
    foreground = rng.rand(200, 3).astype(np.float32) * 0.3
    background = rng.rand(300, 3).astype(np.float32) * 0.3 + 2.0
    scene = pcl.PointCloud(np.vstack([foreground, background]))

    mincut = scene.make_MinCutSegmentation()
    mincut.set_ForegroundPoints(
        pcl.PointCloud(np.array([[0.15, 0.15, 0.15]], dtype=np.float32)))
    mincut.set_Radius(1.0)
    mincut.set_Sigma(0.25)
    mincut.set_SourceWeight(0.8)
    mincut.set_NumberOfNeighbours(14)
    clusters = mincut.extract()

    # PCL returns background first, then foreground.
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [200, 300]
    assert mincut.get_MaxFlow() > 0


def test_min_cut_settings_roundtrip():
    cloud = pcl.PointCloud(
        np.random.RandomState(5).rand(100, 3).astype(np.float32))
    mincut = cloud.make_MinCutSegmentation()
    mincut.set_Radius(2.5)
    mincut.set_Sigma(0.4)
    mincut.set_SourceWeight(0.7)
    mincut.set_NumberOfNeighbours(9)
    assert mincut.get_Radius() == pytest.approx(2.5)
    assert mincut.get_Sigma() == pytest.approx(0.4)
    assert mincut.get_SourceWeight() == pytest.approx(0.7)
    assert mincut.get_NumberOfNeighbours() == 9
