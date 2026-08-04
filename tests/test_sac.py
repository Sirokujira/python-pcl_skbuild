"""Runtime tests for the RANSAC wrappers.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (pcl/pxi/SampleConsensus/).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

PLANE_POINTS = 500
OUTLIER_POINTS = 60


@pytest.fixture
def plane_with_outliers():
    rng = np.random.RandomState(0)
    plane = np.zeros((PLANE_POINTS, 3), dtype=np.float32)
    plane[:, :2] = rng.rand(PLANE_POINTS, 2)
    outliers = rng.rand(OUTLIER_POINTS, 3).astype(np.float32)
    outliers[:, 2] += 2.0
    return pcl.PointCloud(np.vstack([plane, outliers]))


@pytest.fixture
def unit_sphere():
    rng = np.random.RandomState(1)
    theta = rng.rand(400) * 2 * np.pi
    phi = np.arccos(2 * rng.rand(400) - 1)
    points = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ]).astype(np.float32)
    return pcl.PointCloud(points)


def test_ransac_finds_the_plane(plane_with_outliers):
    ransac = pcl.RandomSampleConsensus(
        pcl.SampleConsensusModelPlane(plane_with_outliers))
    ransac.set_DistanceThreshold(0.01)
    assert ransac.computeModel() is True

    inliers = ransac.get_Inliers()
    assert len(inliers) == PLANE_POINTS
    assert max(inliers) < PLANE_POINTS

    coefficients = ransac.get_ModelCoefficients()
    assert len(coefficients) == 4
    # Plane normal is z, up to sign.
    assert abs(coefficients[2]) == pytest.approx(1.0, abs=1e-3)


def test_ransac_finds_the_sphere(unit_sphere):
    ransac = pcl.RandomSampleConsensus(
        pcl.SampleConsensusModelSphere(unit_sphere))
    ransac.set_DistanceThreshold(0.01)
    assert ransac.computeModel() is True

    coefficients = ransac.get_ModelCoefficients()
    # centre(3) + radius.
    assert len(coefficients) == 4
    assert coefficients[3] == pytest.approx(1.0, abs=0.01)
    assert coefficients[:3] == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    assert len(ransac.get_Inliers()) == unit_sphere.size


def test_ransac_inliers_feed_extract_indices(plane_with_outliers):
    ransac = pcl.RandomSampleConsensus(
        pcl.SampleConsensusModelPlane(plane_with_outliers))
    ransac.set_DistanceThreshold(0.01)
    ransac.computeModel()

    extract = plane_with_outliers.make_ExtractIndices()
    extract.set_indices(ransac.get_Inliers())
    assert extract.filter().size == PLANE_POINTS


def test_ransac_settings_roundtrip(plane_with_outliers):
    ransac = pcl.RandomSampleConsensus(
        pcl.SampleConsensusModelPlane(plane_with_outliers))
    ransac.set_DistanceThreshold(0.05)
    ransac.set_MaxIterations(123)
    ransac.set_Probability(0.95)
    assert ransac.get_DistanceThreshold() == pytest.approx(0.05)
    assert ransac.get_MaxIterations() == 123
    assert ransac.get_Probability() == pytest.approx(0.95)


def test_tighter_threshold_keeps_fewer_inliers():
    """A cloud with graded noise: the threshold has to actually bite."""
    rng = np.random.RandomState(2)
    points = np.zeros((600, 3), dtype=np.float32)
    points[:, :2] = rng.rand(600, 2)
    points[:, 2] = rng.uniform(-0.2, 0.2, 600)
    cloud = pcl.PointCloud(points)

    counts = []
    for threshold in (0.02, 0.15):
        ransac = pcl.RandomSampleConsensus(
            pcl.SampleConsensusModelPlane(cloud))
        ransac.set_DistanceThreshold(threshold)
        ransac.computeModel()
        counts.append(len(ransac.get_Inliers()))
    assert counts[0] < counts[1]


@pytest.mark.parametrize("factory", [
    "SampleConsensusModelPlane",
    "SampleConsensusModelLine",
    "SampleConsensusModelCircle2D",
    "SampleConsensusModelCircle3D",
    "SampleConsensusModelSphere",
    "SampleConsensusModelStick",
])
def test_every_wrapped_model_builds_and_runs(plane_with_outliers, factory):
    model = getattr(pcl, factory)(plane_with_outliers)
    assert isinstance(model, pcl.SampleConsensusModel)
    ransac = pcl.RandomSampleConsensus(model)
    ransac.set_DistanceThreshold(0.05)
    ransac.set_MaxIterations(100)
    assert isinstance(ransac.computeModel(), bool)


def test_model_type_is_readable(plane_with_outliers):
    model = pcl.SampleConsensusModelPlane(plane_with_outliers)
    assert model.model_type == pcl.SACMODEL_PLANE


def test_unwrapped_model_type_names_what_is_available(plane_with_outliers):
    with pytest.raises(ValueError, match="SACMODEL_PLANE"):
        pcl.SampleConsensusModel(plane_with_outliers, pcl.SACMODEL_TORUS)


def test_ransac_keeps_its_model_alive(plane_with_outliers):
    """PCL holds the model by shared_ptr, but the Python object owning
    that handle has to outlive the fit too."""
    ransac = pcl.RandomSampleConsensus(
        pcl.SampleConsensusModelPlane(plane_with_outliers))
    ransac.set_DistanceThreshold(0.01)
    assert ransac.computeModel() is True
    assert len(ransac.get_Inliers()) == PLANE_POINTS
