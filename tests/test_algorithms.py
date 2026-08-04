"""Runtime tests for the filter / kdtree / segmentation wrappers.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API under test follows sirokujira/python-pcl (pcl/pxi/Filters/,
pcl/pxi/KdTree/, pcl/pxi/Segmentation/).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def random_cloud():
    """2000 points uniformly filling the unit cube, deterministic."""
    rng = np.random.default_rng(0)
    return pcl.PointCloud(rng.random((2000, 3)).astype(np.float32))


@pytest.fixture
def plane_with_outliers():
    """500 points on z=0 plus 50 well separated points around z=2."""
    rng = np.random.default_rng(1)
    plane = np.zeros((500, 3), dtype=np.float32)
    plane[:, :2] = rng.random((500, 2))
    outliers = rng.random((50, 3)).astype(np.float32)
    outliers[:, 2] += 2.0
    return pcl.PointCloud(np.vstack([plane, outliers]))


# --- filters ---------------------------------------------------------

def test_voxel_grid_reduces_point_count(random_cloud):
    vg = random_cloud.make_voxel_grid_filter()
    vg.set_leaf_size(0.1, 0.1, 0.1)
    out = vg.filter()
    assert isinstance(out, pcl.PointCloud)
    assert 0 < out.size < random_cloud.size


def test_voxel_grid_bigger_leaf_keeps_fewer_points(random_cloud):
    counts = []
    for leaf in (0.05, 0.2):
        vg = random_cloud.make_voxel_grid_filter()
        vg.set_leaf_size(leaf, leaf, leaf)
        counts.append(vg.filter().size)
    assert counts[0] > counts[1]


def test_voxel_grid_direct_construction(random_cloud):
    vg = pcl.VoxelGridFilter(random_cloud)
    vg.set_leaf_size(0.1, 0.1, 0.1)
    assert vg.filter().size > 0


def test_approximate_voxel_grid(random_cloud):
    avg = random_cloud.make_ApproximateVoxelGrid()
    avg.set_leaf_size(0.1, 0.1, 0.1)
    assert 0 < avg.filter().size < random_cloud.size


def test_passthrough_keeps_only_the_range(random_cloud):
    pt = random_cloud.make_passthrough_filter()
    pt.set_filter_field_name("z")
    pt.set_filter_limits(0.0, 0.5)
    out = pt.filter().to_array()
    assert out.shape[0] > 0
    assert out[:, 2].max() <= 0.5


def test_passthrough_negative_inverts_the_selection(random_cloud):
    kept, dropped = [], []
    for negative, sink in ((False, kept), (True, dropped)):
        pt = random_cloud.make_passthrough_filter()
        pt.set_filter_field_name("z")
        pt.set_filter_limits(0.0, 0.5)
        pt.set_negative(negative)
        sink.append(pt.filter().size)
    assert kept[0] + dropped[0] == random_cloud.size


def test_passthrough_field_name_roundtrip(random_cloud):
    pt = random_cloud.make_passthrough_filter()
    pt.set_filter_field_name("y")
    assert pt.get_filter_field_name() == "y"


def test_statistical_outlier_removal(plane_with_outliers):
    so = plane_with_outliers.make_statistical_outlier_filter()
    so.set_mean_k(20)
    so.set_std_dev_mul_thresh(1.0)
    out = so.filter()
    assert 0 < out.size <= plane_with_outliers.size


def test_statistical_outlier_properties(random_cloud):
    so = random_cloud.make_statistical_outlier_filter()
    so.mean_k = 12
    so.stddev_mul_thresh = 2.5
    so.negative = True
    assert so.mean_k == 12
    assert so.stddev_mul_thresh == pytest.approx(2.5)
    assert so.negative is True


def test_radius_outlier_removal_drops_isolated_points(plane_with_outliers):
    ro = plane_with_outliers.make_RadiusOutlierRemoval()
    ro.set_radius_search(0.1)
    ro.set_MinNeighborsInRadius(5)
    out = ro.filter()
    assert out.size < plane_with_outliers.size
    assert ro.get_radius_search() == pytest.approx(0.1)
    assert ro.get_MinNeighborsInRadius() == 5


# --- kdtree ----------------------------------------------------------

def test_kdtree_nearest_k_for_cloud_finds_itself(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    ind, sqdist = kd.nearest_k_search_for_cloud(random_cloud, 3)
    assert ind.shape == (random_cloud.size, 3)
    assert sqdist.shape == (random_cloud.size, 3)
    # Every point's own nearest neighbour is itself, at distance 0.
    assert (ind[:, 0] == np.arange(random_cloud.size)).all()
    assert sqdist[:, 0] == pytest.approx(0.0, abs=1e-6)


def test_kdtree_distances_are_sorted(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    _, sqdist = kd.nearest_k_search_for_cloud(random_cloud, 5)
    assert (np.diff(sqdist, axis=1) >= -1e-6).all()


def test_kdtree_nearest_k_for_point(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    ind, sqdist = kd.nearest_k_search_for_point(random_cloud, 10, 4)
    assert ind.shape == (4,)
    assert ind[0] == 10
    assert sqdist[0] == pytest.approx(0.0, abs=1e-6)


def test_kdtree_negative_index(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    ind, _ = kd.nearest_k_search_for_point(random_cloud, -1, 1)
    assert ind[0] == random_cloud.size - 1


def test_kdtree_index_out_of_range(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    with pytest.raises(IndexError):
        kd.nearest_k_search_for_point(random_cloud, random_cloud.size, 1)


def test_kdtree_rejects_bad_k(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    with pytest.raises(ValueError):
        kd.nearest_k_search_for_cloud(random_cloud, 0)


def test_kdtree_radius_search(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    ind, sqdist = kd.radius_search_for_cloud(random_cloud, 0.1, 16)
    assert ind.shape == (random_cloud.size, 16)
    assert (sqdist[:, 0] <= 0.1 ** 2 + 1e-6).all()


def test_kdtree_radius_search_requires_max_nn(random_cloud):
    kd = random_cloud.make_kdtree_flann()
    with pytest.raises(ValueError):
        kd.radius_search_for_cloud(random_cloud, 0.1)


# --- segmentation ----------------------------------------------------

def test_segmenter_finds_the_plane(plane_with_outliers):
    seg = plane_with_outliers.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    indices, coefficients = seg.segment()

    # The 500 planar points are the inliers; the 50 lifted ones are not.
    assert len(indices) == 500
    assert max(indices) < 500
    # Plane normal is z, so |c| dominates in ax+by+cz+d=0.
    assert len(coefficients) == 4
    assert abs(coefficients[2]) == pytest.approx(1.0, abs=1e-3)


def test_segmenter_returns_plain_lists(plane_with_outliers):
    seg = plane_with_outliers.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    indices, coefficients = seg.segment()
    assert isinstance(indices, list)
    assert isinstance(coefficients, list)
    assert all(isinstance(i, int) for i in indices[:10])


def test_segmenter_max_iterations_accepted(plane_with_outliers):
    seg = plane_with_outliers.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    seg.set_MaxIterations(50)
    indices, _ = seg.segment()
    assert len(indices) > 0


def test_euclidean_cluster_extraction(plane_with_outliers):
    ec = plane_with_outliers.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.1)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)
    clusters = ec.Extract()
    assert len(clusters) >= 1
    assert all(isinstance(c, list) for c in clusters)
    # The dense plane forms one cluster of all 500 of its points.
    assert max(len(c) for c in clusters) == 500


def test_euclidean_cluster_min_size_filters_small_clusters(
        plane_with_outliers):
    ec = plane_with_outliers.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.1)
    ec.set_MinClusterSize(100000)
    ec.set_MaxClusterSize(1000000)
    assert ec.Extract() == []


def test_sac_constants_exposed():
    assert pcl.SAC_RANSAC == 0
    assert pcl.SACMODEL_PLANE == 0
    assert pcl.SACMODEL_LINE == 1
    assert pcl.SACMODEL_SPHERE == 4
