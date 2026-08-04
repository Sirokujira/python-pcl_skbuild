"""Runtime tests for IntegralImageNormalEstimation and
ConditionalEuclideanClustering.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

Both reach PCL through a shim: one for an enum nested in a class
template, one for a `std::function` condition Cython cannot build.
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

GRID = 60


@pytest.fixture
def flat_depth_image():
    """A 60x60 organized cloud at z=1 — normals must all point along z."""
    grid = np.zeros((GRID, GRID, 3), dtype=np.float32)
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    grid[:, :, 0] = cols * 0.05
    grid[:, :, 1] = rows * 0.05
    grid[:, :, 2] = 1.0
    cloud = pcl.PointCloud()
    cloud.from_organized_array(grid)
    return cloud


@pytest.fixture
def two_slabs():
    """300 points near z=0 and 300 near z=0.5, interleaved in space so a
    plain distance test would merge them."""
    rng = np.random.RandomState(0)
    lower = rng.rand(300, 3).astype(np.float32)
    lower[:, 2] *= 0.02
    upper = rng.rand(300, 3).astype(np.float32)
    upper[:, 2] = upper[:, 2] * 0.02 + 0.5
    return pcl.PointCloud(np.vstack([lower, upper]))


# --- integral image normals -------------------------------------------

def test_integral_image_normals_on_a_flat_image(flat_depth_image):
    estimator = flat_depth_image.make_IntegralImageNormalEstimation()
    estimator.set_NormalEstimationMethod(pcl.AVERAGE_3D_GRADIENT)
    estimator.set_MaxDepthChangeFactor(0.02)
    estimator.set_NormalSmoothingSize(10.0)
    normals = estimator.compute()

    assert normals.shape == (flat_depth_image.size, 4)
    finite = normals[np.isfinite(normals[:, 2])]
    assert len(finite) > 0
    assert np.abs(finite[:, 2]) == pytest.approx(
        np.ones(len(finite)), abs=1e-3)


def test_integral_image_normals_as_a_cloud(flat_depth_image):
    estimator = flat_depth_image.make_IntegralImageNormalEstimation()
    estimator.set_NormalEstimationMethod(pcl.AVERAGE_3D_GRADIENT)
    normals = estimator.compute_cloud()
    assert isinstance(normals, pcl.PointCloud_Normal)
    assert normals.size == flat_depth_image.size


@pytest.mark.parametrize("method", [
    pcl.COVARIANCE_MATRIX,
    pcl.AVERAGE_3D_GRADIENT,
    pcl.AVERAGE_DEPTH_CHANGE,
    pcl.SIMPLE_3D_GRADIENT,
])
def test_every_estimation_method_runs(flat_depth_image, method):
    estimator = flat_depth_image.make_IntegralImageNormalEstimation()
    estimator.set_NormalEstimationMethod(method)
    estimator.set_MaxDepthChangeFactor(0.02)
    estimator.set_NormalSmoothingSize(10.0)
    assert estimator.compute().shape[1] == 4


def test_estimation_method_constants_match_pcls_enum():
    """Taken from the header through the shim, not copied into Python."""
    assert (pcl.COVARIANCE_MATRIX, pcl.AVERAGE_3D_GRADIENT,
            pcl.AVERAGE_DEPTH_CHANGE, pcl.SIMPLE_3D_GRADIENT) == (0, 1, 2, 3)


def test_integral_image_normals_reject_an_unorganized_cloud():
    """PCL returns nothing rather than complaining, which reads as a bug
    in the caller's code."""
    cloud = pcl.PointCloud(
        np.random.RandomState(1).rand(100, 3).astype(np.float32))
    with pytest.raises(ValueError, match="organized"):
        cloud.make_IntegralImageNormalEstimation()


# --- conditional euclidean clustering ---------------------------------

def test_condition_separates_what_distance_alone_would_merge(two_slabs):
    """Tolerance 0.6 spans both slabs; only the predicate keeps them
    apart."""
    clustering = two_slabs.make_ConditionalEuclideanClustering()
    clustering.set_ClusterTolerance(0.6)
    clustering.set_MinClusterSize(50)
    clustering.set_MaxClusterSize(10000)
    clustering.set_ConditionFunction(
        lambda a, b, sqr_distance: abs(a[2] - b[2]) < 0.05)

    clusters = clustering.segment()
    assert sorted(len(c) for c in clusters) == [300, 300]


def test_condition_receives_points_and_a_distance(two_slabs):
    seen = []

    def record(a, b, sqr_distance):
        seen.append((a, b, sqr_distance))
        return True

    clustering = two_slabs.make_ConditionalEuclideanClustering()
    clustering.set_ClusterTolerance(0.1)
    clustering.set_MinClusterSize(1)
    clustering.set_ConditionFunction(record)
    clustering.segment()

    assert seen
    a, b, sqr_distance = seen[0]
    assert len(a) == 3 and len(b) == 3
    assert sqr_distance >= 0.0


def test_raising_condition_does_not_kill_the_process(two_slabs, capfd):
    """An exception escaping into a noexcept C callback would terminate
    the interpreter; it must be printed and treated as "not joined"."""
    def explode(a, b, sqr_distance):
        raise ValueError("boom")

    clustering = two_slabs.make_ConditionalEuclideanClustering()
    clustering.set_ClusterTolerance(0.1)
    clustering.set_MinClusterSize(1)
    clustering.set_ConditionFunction(explode)
    clustering.segment()

    assert "ValueError" in capfd.readouterr().err


def test_segment_without_a_condition_reports_it(two_slabs):
    clustering = two_slabs.make_ConditionalEuclideanClustering()
    clustering.set_ClusterTolerance(0.1)
    with pytest.raises(RuntimeError, match="set_ConditionFunction"):
        clustering.segment()


def test_condition_must_be_callable(two_slabs):
    clustering = two_slabs.make_ConditionalEuclideanClustering()
    with pytest.raises(TypeError):
        clustering.set_ConditionFunction("not callable")


def test_clustering_settings_roundtrip(two_slabs):
    clustering = two_slabs.make_ConditionalEuclideanClustering()
    clustering.set_ClusterTolerance(0.25)
    clustering.set_MinClusterSize(7)
    clustering.set_MaxClusterSize(999)
    assert clustering.get_ClusterTolerance() == pytest.approx(0.25)
    assert clustering.get_MinClusterSize() == 7
    assert clustering.get_MaxClusterSize() == 999


def test_always_true_condition_matches_plain_euclidean(two_slabs):
    """With a predicate that never splits, the result has to agree with
    EuclideanClusterExtraction on the same tolerance."""
    conditional = two_slabs.make_ConditionalEuclideanClustering()
    conditional.set_ClusterTolerance(0.1)
    conditional.set_MinClusterSize(10)
    conditional.set_MaxClusterSize(10000)
    conditional.set_ConditionFunction(lambda a, b, d: True)

    plain = two_slabs.make_EuclideanClusterExtraction()
    plain.set_ClusterTolerance(0.1)
    plain.set_MinClusterSize(10)
    plain.set_MaxClusterSize(10000)

    assert (sorted(len(c) for c in conditional.segment())
            == sorted(len(c) for c in plain.Extract()))
