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


# --- global registration (no initial guess) ----------------------------

def _l_shaped(n=400, seed=1):
    """An asymmetric slab: a shape whose alignment onto a copy of itself
    is unique, so a recovered pose can be checked against the truth."""
    rng = np.random.RandomState(seed)
    base = rng.rand(int(n * 0.67), 3).astype(np.float32) * [1.0, 0.6, 0.1]
    arm = (rng.rand(n - int(n * 0.67), 3).astype(np.float32)
           * [0.3, 0.6, 0.4] + [0.0, 0.0, 0.1])
    return np.vstack([base, arm])


def _far_apart():
    """A pose far outside any basin ICP could be started in: 1.1 rad and
    ~4 units away."""
    points = _l_shaped()
    angle = 1.1
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([3.0, -2.0, 1.5])
    source = pcl.PointCloud(points)
    target = pcl.PointCloud((points @ rotation.T + translation).astype(
        np.float32))
    return source, target, rotation, translation


def _fpfh(cloud):
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(15)
    estimator = cloud.make_FPFHEstimation()
    estimator.set_InputNormals(normals.compute_cloud())
    estimator.set_RadiusSearch(0.12)
    return estimator.compute()


def _aligner(cloud):
    align = cloud.make_SampleConsensusPrerejective()
    align.set_MaxCorrespondenceDistance(0.05)
    align.set_InlierFraction(0.25)
    return align


def test_global_alignment_needs_no_initial_guess():
    """The gap this fills: ICP, ICP-NL, GICP and NDT all refine a pose
    that is already roughly right."""
    source, target, rotation, translation = _far_apart()
    converged, matrix, _, _ = _aligner(source).align(
        source, _fpfh(source), target, _fpfh(target))

    assert converged
    assert matrix[:3, 3] == pytest.approx(translation, abs=0.05)
    assert matrix[:3, :3] == pytest.approx(rotation, abs=0.05)


def test_global_alignment_returns_icps_shape():
    source, target, _, _ = _far_apart()
    result = _aligner(source).align(
        source, _fpfh(source), target, _fpfh(target))

    assert len(result) == 4
    converged, matrix, estimate, fitness = result
    assert isinstance(converged, bool)
    assert matrix.shape == (4, 4)
    assert isinstance(estimate, pcl.PointCloud)
    assert estimate.size == source.size
    assert fitness >= 0.0


def test_global_estimate_matches_applying_the_transform():
    source, target, _, _ = _far_apart()
    _, matrix, estimate, _ = _aligner(source).align(
        source, _fpfh(source), target, _fpfh(target))
    assert source.transform(matrix).to_array() == pytest.approx(
        estimate.to_array(), abs=1e-4)


def test_global_inliers_index_the_source():
    source, target, _, _ = _far_apart()
    align = _aligner(source)
    assert align.get_Inliers() is None

    align.align(source, _fpfh(source), target, _fpfh(target))
    inliers = align.get_Inliers()
    assert inliers
    assert max(inliers) < source.size
    assert min(inliers) >= 0


def test_icp_refines_a_global_result_to_convergence():
    """The documented next step: global gets ICP into its basin."""
    source, target, _, _ = _far_apart()
    _, matrix, _, _ = _aligner(source).align(
        source, _fpfh(source), target, _fpfh(target))

    icp = source.make_IterativeClosestPoint()
    converged, _, _, fitness = icp.icp(source.transform(matrix), target)
    assert converged
    assert fitness < 1e-6


def test_global_alignment_settings_roundtrip():
    align = pcl.SampleConsensusPrerejective()
    align.set_MaximumIterations(1234)
    align.set_NumberOfSamples(4)
    align.set_CorrespondenceRandomness(7)
    align.set_SimilarityThreshold(0.8)
    align.set_MaxCorrespondenceDistance(0.02)
    align.set_InlierFraction(0.5)

    assert align.get_MaximumIterations() == 1234
    assert align.get_NumberOfSamples() == 4
    assert align.get_CorrespondenceRandomness() == 7
    assert align.get_SimilarityThreshold() == pytest.approx(0.8)
    assert align.get_MaxCorrespondenceDistance() == pytest.approx(0.02)
    assert align.get_InlierFraction() == pytest.approx(0.5)


def test_global_alignment_rejects_nonsense_settings():
    align = pcl.SampleConsensusPrerejective()
    with pytest.raises(ValueError, match="6-DOF"):
        align.set_NumberOfSamples(2)
    with pytest.raises(ValueError, match="inlier_fraction"):
        align.set_InlierFraction(0.0)
    with pytest.raises(ValueError, match="similarity_threshold"):
        align.set_SimilarityThreshold(1.0)
    with pytest.raises(ValueError, match="max_correspondence_distance"):
        align.set_MaxCorrespondenceDistance(0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        align.set_MaximumIterations(0)


def test_global_alignment_checks_the_descriptor_arrays():
    source, target, _, _ = _far_apart()
    source_desc, target_desc = _fpfh(source), _fpfh(target)
    align = _aligner(source)

    with pytest.raises(ValueError, match="one row per point"):
        align.align(source, source_desc[:10], target, target_desc)
    with pytest.raises(ValueError, match=r"\(n, 33\)"):
        align.align(source,
                    np.zeros((source.size, 20), dtype=np.float32),
                    target, target_desc)
