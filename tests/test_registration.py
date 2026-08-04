"""Runtime tests for the registration wrappers.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

Each algorithm gets the same job: recover a known rotation. The API
follows sirokujira/python-pcl (pcl/pxi/registration/), except
NormalDistributionsTransform, which python-pcl never had
(strawlab/python-pcl#265).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


@pytest.fixture
def pair():
    """A cloud and a mildly rotated copy of it."""
    points = np.random.RandomState(42).rand(900, 3).astype(np.float32)
    rotated = (points @ rotation_z(0.05).T).astype(np.float32)
    return pcl.PointCloud(points), pcl.PointCloud(rotated)


def check_result(converged, transform, estimate, fitness, source):
    assert converged is True
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float32
    # Eigen is column-major and the wrapper hands the buffer over as-is.
    assert transform.flags["F_CONTIGUOUS"]
    # Bottom row of a rigid transform.
    assert transform[3, :] == pytest.approx([0, 0, 0, 1], abs=1e-5)
    assert isinstance(estimate, pcl.PointCloud)
    assert estimate.size == source.size
    assert fitness < 0.1


def test_icp_recovers_the_rotation(pair):
    source, target = pair
    icp = source.make_IterativeClosestPoint()
    check_result(*icp.icp(source, target, max_iter=100), source=source)


def test_icp_nl_recovers_the_rotation(pair):
    source, target = pair
    icp = source.make_IterativeClosestPointNonLinear()
    check_result(*icp.icp_nl(source, target, max_iter=100), source=source)


def test_gicp_recovers_the_rotation(pair):
    source, target = pair
    gicp = source.make_GeneralizedIterativeClosestPoint()
    check_result(*gicp.gicp(source, target, max_iter=100), source=source)


def test_ndt_recovers_the_rotation(pair):
    source, target = pair
    ndt = source.make_NormalDistributionsTransform()
    ndt.set_Resolution(1.0)
    ndt.set_StepSize(0.1)
    ndt.set_TransformationEpsilon(0.01)
    check_result(*ndt.ndt(source, target, max_iter=35), source=source)


def test_icp_transform_moves_source_onto_target(pair):
    """The returned matrix is the transform, not decoration: applying it
    to the source has to reproduce the estimate."""
    source, target = pair
    icp = source.make_IterativeClosestPoint()
    _, transform, estimate, _ = icp.icp(source, target, max_iter=100)

    points = source.to_array()
    homogeneous = np.hstack([points, np.ones((len(points), 1), np.float32)])
    applied = (homogeneous @ np.asarray(transform).T)[:, :3]
    assert applied == pytest.approx(estimate.to_array(), abs=1e-4)


def test_ndt_reports_its_own_diagnostics(pair):
    source, target = pair
    ndt = source.make_NormalDistributionsTransform()
    ndt.set_Resolution(1.0)
    ndt.set_StepSize(0.1)
    ndt.ndt(source, target, max_iter=35)
    assert ndt.get_FinalNumIteration() >= 1
    assert ndt.get_TransformationProbability() > 0.0
    assert ndt.get_Resolution() == pytest.approx(1.0)
    assert ndt.get_StepSize() == pytest.approx(0.1)


def test_registration_settings_are_accepted(pair):
    source, target = pair
    icp = source.make_IterativeClosestPoint()
    icp.set_MaximumIterations(10)
    icp.set_MaxCorrespondenceDistance(0.5)
    icp.set_TransformationEpsilon(1e-8)
    icp.set_EuclideanFitnessEpsilon(1e-6)
    icp.set_UseReciprocalCorrespondences(True)
    converged, _, _, _ = icp.icp(source, target)
    assert isinstance(converged, bool)


def test_gicp_specific_settings(pair):
    source, target = pair
    gicp = source.make_GeneralizedIterativeClosestPoint()
    gicp.set_RotationEpsilon(1e-4)
    gicp.set_CorrespondenceRandomness(15)
    gicp.set_MaximumOptimizerIterations(15)
    converged, _, _, _ = gicp.gicp(source, target, max_iter=50)
    assert converged is True


def test_identical_clouds_give_an_identity_transform():
    points = np.random.RandomState(7).rand(300, 3).astype(np.float32)
    cloud = pcl.PointCloud(points)
    icp = cloud.make_IterativeClosestPoint()
    _, transform, _, fitness = icp.icp(cloud, cloud, max_iter=50)
    assert np.asarray(transform) == pytest.approx(np.eye(4), abs=1e-4)
    assert fitness == pytest.approx(0.0, abs=1e-6)
