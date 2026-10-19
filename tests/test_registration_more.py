"""Runtime tests for the second registration batch: point-to-plane ICP,
SAC-IA, (K)FPCS, and the correspondence pipeline exposed as functions.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _matrix(rotation, translation):
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _l_shaped(n=600):
    rng = np.random.RandomState(7)
    base = rng.rand(int(n * 0.67), 3).astype(np.float32) * [1.0, 0.6, 0.1]
    arm = (rng.rand(n - int(n * 0.67), 3).astype(np.float32)
           * [0.3, 0.6, 0.4] + [0.0, 0.0, 0.1])
    return np.vstack([base, arm])


def _oriented(points):
    """A PointNormal cloud with estimated normals."""
    cloud = pcl.PointCloud(points)
    estimator = cloud.make_NormalEstimation()
    estimator.set_KSearch(15)
    return pcl.PointCloud_PointNormal().from_xyz_and_normals(
        cloud, estimator.compute_cloud())


def _fpfh(cloud):
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(15)
    estimator = cloud.make_FPFHEstimation()
    estimator.set_InputNormals(normals.compute_cloud())
    estimator.set_RadiusSearch(0.12)
    return estimator.compute()


@pytest.fixture
def moved_pair():
    """An L-shaped cloud and a copy moved well outside ICP's basin."""
    points = _l_shaped()
    rotation = _rotation_z(0.9)
    translation = np.array([2.0, -1.5, 1.0], dtype=np.float32)
    source = pcl.PointCloud(points)
    target = pcl.PointCloud((points @ rotation.T + translation).astype(
        np.float32))
    return source, target, _matrix(rotation, translation)


# --- IterativeClosestPointWithNormals -----------------------------------


def test_icp_with_normals_recovers_a_small_rotation():
    points = _l_shaped()
    source = _oriented(points)
    target = _oriented((points @ _rotation_z(0.05).T).astype(np.float32))
    icp = source.make_IterativeClosestPointWithNormals()
    converged, transform, estimate, fitness = icp.icp(
        source, target, max_iter=50)

    assert converged is True
    assert transform.shape == (4, 4)
    assert transform[3, :] == pytest.approx([0, 0, 0, 1], abs=1e-5)
    assert transform[:3, :3] == pytest.approx(_rotation_z(0.05), abs=0.02)
    assert isinstance(estimate, pcl.PointCloud_PointNormal)
    assert estimate.size == source.size
    assert fitness < 0.01


def test_icp_with_normals_estimate_is_source_transformed():
    points = _l_shaped()
    source = _oriented(points)
    target = _oriented((points @ _rotation_z(0.03).T).astype(np.float32))
    icp = source.make_IterativeClosestPointWithNormals()
    _, transform, estimate, _ = icp.icp(source, target, max_iter=30)
    assert estimate.to_array()[:, :3] == pytest.approx(
        source.transform(transform).to_array()[:, :3], abs=1e-4)


def test_icp_with_normals_settings_round_trip():
    icp = pcl.IterativeClosestPointWithNormals()
    assert icp.get_UseSymmetricObjective() is False
    icp.set_UseSymmetricObjective(True)
    assert icp.get_UseSymmetricObjective() is True
    icp.set_EnforceSameDirectionNormals(False)
    assert icp.get_EnforceSameDirectionNormals() is False


def test_icp_with_normals_rejects_empty_clouds():
    icp = pcl.IterativeClosestPointWithNormals()
    with pytest.raises(ValueError):
        icp.icp(pcl.PointCloud_PointNormal(), _oriented(_l_shaped()))


# --- SampleConsensusInitialAlignment ------------------------------------


def test_sac_ia_needs_no_initial_guess(moved_pair):
    source, target, truth = moved_pair
    align = source.make_SampleConsensusInitialAlignment()
    align.set_MaxCorrespondenceDistance(0.05)
    align.set_MaximumIterations(1000)
    converged, transform, estimate, fitness = align.align(
        source, _fpfh(source), target, _fpfh(target))

    assert isinstance(converged, bool)
    assert transform.shape == (4, 4)
    assert isinstance(estimate, pcl.PointCloud)
    assert estimate.size == source.size
    # Far better than doing nothing, even if not pixel-perfect.
    before = pcl.validate_transformation(source, target, np.eye(4))
    after = pcl.validate_transformation(source, target, transform)
    assert after < before / 10
    assert transform[:3, 3] == pytest.approx(truth[:3, 3], abs=0.1)


def test_sac_ia_settings_round_trip():
    align = pcl.SampleConsensusInitialAlignment()
    align.set_MinSampleDistance(0.02)
    assert align.get_MinSampleDistance() == pytest.approx(0.02)
    align.set_MaxCorrespondenceDistance(0.3)
    assert align.get_MaxCorrespondenceDistance() == pytest.approx(0.3)
    align.set_MaximumIterations(42)
    assert align.get_MaximumIterations() == 42
    align.set_NumberOfSamples(4)
    assert align.get_NumberOfSamples() == 4
    align.set_CorrespondenceRandomness(7)
    assert align.get_CorrespondenceRandomness() == 7


def test_sac_ia_rejects_mismatched_descriptors(moved_pair):
    source, target, _ = moved_pair
    align = source.make_SampleConsensusInitialAlignment()
    with pytest.raises(ValueError):
        align.align(source, np.zeros((3, 33), np.float32), target,
                    _fpfh(target))
    with pytest.raises(ValueError):
        align.align(source, np.zeros((source.size, 5), np.float32), target,
                    _fpfh(target))


# --- FPCS / KFPCS --------------------------------------------------------


def _fpcs_settings(align):
    align.set_ApproxOverlap(0.9)
    align.set_Delta(0.02)
    align.set_NumberOfSamples(200)
    align.set_MaxComputationTime(30)
    align.set_NumberOfThreads(1)


def test_fpcs_aligns_without_descriptors(moved_pair):
    source, target, truth = moved_pair
    align = source.make_FPCSInitialAlignment()
    _fpcs_settings(align)
    converged, transform, estimate, fitness = align.align(source, target)

    assert isinstance(converged, bool)
    assert transform.shape == (4, 4)
    assert estimate.size == source.size
    before = pcl.validate_transformation(source, target, np.eye(4))
    after = pcl.validate_transformation(source, target, transform)
    assert after < before / 10


def test_kfpcs_aligns_without_descriptors(moved_pair):
    source, target, truth = moved_pair
    align = source.make_KFPCSInitialAlignment()
    _fpcs_settings(align)
    converged, transform, estimate, fitness = align.align(source, target)

    assert transform.shape == (4, 4)
    assert estimate.size == source.size
    before = pcl.validate_transformation(source, target, np.eye(4))
    after = pcl.validate_transformation(source, target, transform)
    assert after < before / 10


def test_fpcs_settings_round_trip():
    align = pcl.FPCSInitialAlignment()
    align.set_Delta(0.5)
    assert align.get_Delta() == pytest.approx(0.5)
    align.set_ApproxOverlap(0.4)
    assert align.get_ApproxOverlap() == pytest.approx(0.4)
    align.set_ScoreThreshold(0.3)
    assert align.get_ScoreThreshold() == pytest.approx(0.3)
    with pytest.raises(ValueError):
        align.set_Delta(0.0)


def test_kfpcs_settings_round_trip():
    align = pcl.KFPCSInitialAlignment()
    align.set_UpperTranslationThreshold(2.0)
    assert align.get_UpperTranslationThreshold() == pytest.approx(2.0)
    align.set_LowerTranslationThreshold(0.5)
    assert align.get_LowerTranslationThreshold() == pytest.approx(0.5)
    align.set_Lambda(0.3)
    assert align.get_Lambda() == pytest.approx(0.3)


def test_fpcs_rejects_tiny_clouds():
    align = pcl.FPCSInitialAlignment()
    with pytest.raises(ValueError):
        align.align(pcl.PointCloud(np.zeros((3, 3), np.float32)),
                    pcl.PointCloud(_l_shaped()))


# --- the correspondence pipeline ----------------------------------------


@pytest.fixture
def jittered_pair():
    """A cloud and a copy shifted by a known offset with tiny noise, so
    point i of the source corresponds to point i of the target."""
    rng = np.random.RandomState(3)
    points = rng.rand(400, 3).astype(np.float32)
    shift = np.array([0.01, -0.02, 0.015], dtype=np.float32)
    noise = (rng.randn(400, 3) * 1e-4).astype(np.float32)
    return (pcl.PointCloud(points), pcl.PointCloud(points + shift + noise),
            shift)


def test_estimate_correspondences_shapes(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    assert pairs.shape == (source.size, 2)
    assert pairs.dtype == np.int32
    assert distances.shape == (source.size,)
    assert distances.dtype == np.float32
    # Source index column is every source point, once.
    assert sorted(pairs[:, 0].tolist()) == list(range(source.size))
    # The shifted copy is the nearest neighbour for nearly every point.
    assert np.mean(pairs[:, 0] == pairs[:, 1]) > 0.9


def test_correspondence_distances_are_squared(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    s = source.to_array()[pairs[:, 0]]
    t = target.to_array()[pairs[:, 1]]
    assert distances == pytest.approx(np.sum((s - t) ** 2, axis=1), abs=1e-6)


def test_estimate_correspondences_max_distance_and_reciprocal(jittered_pair):
    source, target, _ = jittered_pair
    full, _ = pcl.estimate_correspondences(source, target)
    close, dists = pcl.estimate_correspondences(source, target,
                                                max_distance=0.01)
    assert len(close) < len(full)
    assert np.all(dists <= 0.01 ** 2 + 1e-7)
    mutual, _ = pcl.estimate_correspondences(source, target, reciprocal=True)
    assert 0 < len(mutual) <= len(full)
    # Mutual pairs are unique on both sides.
    assert len(set(mutual[:, 1].tolist())) == len(mutual)
    with pytest.raises(ValueError):
        pcl.estimate_correspondences(source, target, max_distance=0)


def test_reject_one_to_one_makes_targets_unique(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    kept, kept_d = pcl.reject_one_to_one(pairs, distances)
    assert len(kept) <= len(pairs)
    assert len(set(kept[:, 1].tolist())) == len(kept)
    assert kept_d.shape == (len(kept),)


def test_reject_by_distance(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    kept, kept_d = pcl.reject_by_distance(pairs, distances, 0.02)
    assert 0 < len(kept) < len(pairs)
    assert np.all(kept_d <= 0.02 ** 2 + 1e-7)
    with pytest.raises(ValueError):
        pcl.reject_by_distance(pairs, distances, 0.0)


def test_reject_median_distance(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    kept, kept_d = pcl.reject_median_distance(pairs, distances, factor=1.0)
    assert 0 < len(kept) < len(pairs)
    # Roughly half survive a factor of 1.0 (median is in squared units).
    assert 0.3 < len(kept) / len(pairs) < 0.7
    with pytest.raises(ValueError):
        pcl.reject_median_distance(pairs, distances, factor=0)


def test_reject_trimmed(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    kept, kept_d = pcl.reject_trimmed(pairs, distances, 0.5)
    assert len(kept) == pytest.approx(len(pairs) * 0.5, abs=2)
    # The kept ones are the closest ones.
    assert kept_d.max() <= np.sort(distances)[len(kept)] + 1e-7
    few, _ = pcl.reject_trimmed(pairs, distances, 0.01,
                                min_correspondences=50)
    assert len(few) == 50
    with pytest.raises(ValueError):
        pcl.reject_trimmed(pairs, distances, 1.5)


def test_reject_sample_consensus_recovers_the_shift(jittered_pair):
    source, target, shift = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    # Corrupt a fifth of the matches with random partners.
    rng = np.random.RandomState(1)
    bad = rng.choice(len(pairs), len(pairs) // 5, replace=False)
    pairs = pairs.copy()
    pairs[bad, 1] = rng.randint(0, target.size, len(bad))
    kept, kept_d, transform = pcl.reject_sample_consensus(
        source, target, pairs, distances, inlier_threshold=0.005,
        max_iterations=500)

    assert len(kept) < len(pairs)
    assert len(kept) > len(pairs) * 0.6
    assert transform.shape == (4, 4)
    assert transform[:3, 3] == pytest.approx(shift, abs=2e-3)
    assert transform[:3, :3] == pytest.approx(np.eye(3), abs=1e-2)
    with pytest.raises(ValueError):
        pcl.reject_sample_consensus(source, target, pairs, distances, 0.0)


def test_reject_surface_normal_drops_flipped_normals(jittered_pair):
    source, target, _ = jittered_pair
    pairs, distances = pcl.estimate_correspondences(source, target)
    up = np.tile([0.0, 0.0, 1.0, 0.0], (source.size, 1)).astype(np.float32)
    flipped = up.copy()
    flipped[: source.size // 2, 2] = -1.0
    kept, _ = pcl.reject_surface_normal(
        source, target, pcl.PointCloud_Normal(up),
        pcl.PointCloud_Normal(up), pairs, distances, threshold=0.9)
    assert len(kept) == len(pairs)
    kept, _ = pcl.reject_surface_normal(
        source, target, pcl.PointCloud_Normal(flipped),
        pcl.PointCloud_Normal(up), pairs, distances, threshold=0.9)
    assert len(kept) == pytest.approx(len(pairs) / 2, abs=25)
    with pytest.raises(ValueError):
        pcl.reject_surface_normal(
            source, target, pcl.PointCloud_Normal(up[:3]),
            pcl.PointCloud_Normal(up), pairs, distances, 0.5)


@pytest.mark.parametrize("method", ["svd", "lm"])
def test_estimate_transformation_recovers_the_shift(jittered_pair, method):
    source, target, shift = jittered_pair
    transform = pcl.estimate_transformation(source, target, method=method)
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float32
    assert transform[:3, 3] == pytest.approx(shift, abs=1e-3)
    assert transform[:3, :3] == pytest.approx(np.eye(3), abs=1e-3)


def test_estimate_transformation_with_pairs(jittered_pair):
    source, target, shift = jittered_pair
    pairs = np.column_stack([np.arange(100), np.arange(100)])
    transform = pcl.estimate_transformation(source, target, pairs)
    assert transform[:3, 3] == pytest.approx(shift, abs=1e-3)
    with pytest.raises(ValueError):
        pcl.estimate_transformation(source, target, pairs[:2])
    with pytest.raises(ValueError):
        pcl.estimate_transformation(source, target, method="ransac")
    with pytest.raises(ValueError):
        pcl.estimate_transformation(
            source, pcl.PointCloud(target.to_array()[:10]))


def test_validate_transformation_scores_the_fit(jittered_pair):
    source, target, shift = jittered_pair
    identity = pcl.validate_transformation(source, target, np.eye(4))
    truth = pcl.validate_transformation(source, target,
                                        _matrix(np.eye(3), shift))
    assert truth < identity
    assert truth == pytest.approx(0.0, abs=1e-6)
    # max_range drops target points farther than that from the moved
    # source; a cutoff below every distance leaves nothing to average.
    ranged = pcl.validate_transformation(source, target, np.eye(4),
                                         max_range=0.1)
    assert ranged <= identity
    with pytest.raises(ValueError):
        pcl.validate_transformation(source, target, np.eye(4), max_range=0)
    with pytest.raises(ValueError):
        pcl.validate_transformation(pcl.PointCloud(), target, np.eye(4))
