"""Runtime tests for the FPFH and SHOT descriptors, and for the full
recognition pipeline they were added to complete: descriptors ->
matching -> correspondence grouping -> pose.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def box():
    """Points on a unit box's surface — varied local geometry, so the
    descriptors have something to describe.

    600 points per face: SHOT refuses a descriptor (and FPFH degrades to
    an empty histogram) for a point whose support radius holds too few
    neighbours, so the surface must be dense relative to the radii the
    tests use.
    """
    rng = np.random.RandomState(0)
    face = rng.rand(600, 2).astype(np.float32)
    points = []
    for u, v in face:
        points += [[u, v, 0.0], [u, v, 1.0], [u, 0.0, v],
                   [u, 1.0, v], [0.0, u, v], [1.0, u, v]]
    return pcl.PointCloud(np.array(points, dtype=np.float32))


@pytest.fixture
def box_normals(box):
    estimator = box.make_NormalEstimation()
    estimator.set_KSearch(15)
    return estimator.compute_cloud()


def _fpfh(cloud, normals, radius=0.08):
    estimator = cloud.make_FPFHEstimation()
    estimator.set_InputNormals(normals)
    estimator.set_RadiusSearch(radius)
    return estimator


def _shot(cloud, normals, radius=0.1):
    estimator = cloud.make_SHOTEstimation()
    estimator.set_InputNormals(normals)
    estimator.set_RadiusSearch(radius)
    return estimator


# --- FPFH --------------------------------------------------------------

def test_fpfh_returns_one_histogram_per_point(box, box_normals):
    descriptors = _fpfh(box, box_normals).compute()
    assert descriptors.shape == (box.size, 33)
    assert descriptors.dtype == np.float32
    assert np.isfinite(descriptors).all()


def test_fpfh_histograms_are_percentages(box, box_normals):
    """Each of the three 11-bin features is normalized to sum to 100, so
    a full histogram sums to 300 — the property matching relies on."""
    descriptors = _fpfh(box, box_normals).compute()
    assert descriptors.sum(axis=1) == pytest.approx(
        np.full(box.size, 300.0), abs=1.0)


def test_fpfh_distinguishes_edge_from_face(box, box_normals):
    """A point on a flat face and a point near an edge see different
    geometry; identical descriptors would be useless for matching."""
    points = box.to_array()
    face_index = int(np.argmin(
        np.abs(points[:, 2]) + np.abs(points[:, 0] - 0.5)
        + np.abs(points[:, 1] - 0.5)))
    edge_index = int(np.argmin(
        np.abs(points[:, 2]) + np.abs(points[:, 0]) + np.abs(points[:, 1])))

    descriptors = _fpfh(box, box_normals).compute()
    assert not np.allclose(descriptors[face_index], descriptors[edge_index],
                           atol=1.0)


def test_fpfh_supports_ksearch_too(box, box_normals):
    estimator = box.make_FPFHEstimation()
    estimator.set_InputNormals(box_normals)
    estimator.set_KSearch(20)
    assert estimator.get_KSearch() == 20
    assert estimator.compute().shape == (box.size, 33)


def test_fpfh_settings_roundtrip(box, box_normals):
    estimator = _fpfh(box, box_normals, radius=0.07)
    assert estimator.get_RadiusSearch() == pytest.approx(0.07)


def test_fpfh_error_paths(box, box_normals):
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.FPFHEstimation().compute()

    estimator = box.make_FPFHEstimation()
    with pytest.raises(RuntimeError, match="normals"):
        estimator.compute()

    estimator.set_InputNormals(box_normals)
    with pytest.raises(RuntimeError, match="KSearch|RadiusSearch"):
        estimator.compute()

    short = pcl.PointCloud_Normal(np.zeros((5, 4), dtype=np.float32))
    estimator.set_InputNormals(short)
    estimator.set_RadiusSearch(0.05)
    with pytest.raises(ValueError, match="one entry per input point"):
        estimator.compute()


# --- SHOT --------------------------------------------------------------

def test_shot_returns_one_descriptor_per_point(box, box_normals):
    descriptors = _shot(box, box_normals).compute()
    assert descriptors.shape == (box.size, 352)
    assert descriptors.dtype == np.float32
    # A dense box surface gives every point a stable reference frame.
    assert np.isfinite(descriptors).all()


def test_shot_descriptors_are_normalized(box, box_normals):
    """SHOT normalizes each descriptor to unit length."""
    descriptors = _shot(box, box_normals).compute()
    norms = np.linalg.norm(descriptors, axis=1)
    assert norms == pytest.approx(np.ones(box.size), abs=1e-3)


def test_shot_lrf_radius_roundtrip(box, box_normals):
    estimator = _shot(box, box_normals)
    estimator.set_LRFRadius(0.15)
    assert estimator.get_LRFRadius() == pytest.approx(0.15)


def test_shot_requires_a_radius(box, box_normals):
    """SHOT rejects a k-nearest setup, so the wrapper never offers one —
    and compute() without a radius reports it instead of returning an
    empty array."""
    estimator = box.make_SHOTEstimation()
    estimator.set_InputNormals(box_normals)
    assert not hasattr(estimator, "set_KSearch")
    with pytest.raises(RuntimeError, match="RadiusSearch"):
        estimator.compute()
    with pytest.raises(ValueError, match="radius"):
        estimator.set_RadiusSearch(0.0)


def test_shot_error_paths(box, box_normals):
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.SHOTEstimation().compute()

    estimator = box.make_SHOTEstimation()
    with pytest.raises(RuntimeError, match="normals"):
        estimator.compute()


# --- the full recognition pipeline -------------------------------------

def test_descriptors_matching_and_grouping_recover_a_pose():
    """The reason these descriptors exist here: FPFH on model and scene,
    nearest-neighbour matching, and GeometricConsistencyGrouping must
    recover a known transform end to end."""
    rng = np.random.RandomState(1)
    # An L-shaped slab: asymmetric, so descriptors are distinctive.
    base = rng.rand(800, 3).astype(np.float32) * [1.0, 0.6, 0.1]
    arm = (rng.rand(400, 3).astype(np.float32) * [0.3, 0.6, 0.4]
           + [0.0, 0.0, 0.1])
    model_points = np.vstack([base, arm])
    translation = np.array([2.0, 1.0, 3.0], dtype=np.float32)

    model = pcl.PointCloud(model_points)
    scene = pcl.PointCloud(model_points + translation)

    def describe(cloud):
        normal_est = cloud.make_NormalEstimation()
        normal_est.set_KSearch(15)
        est = cloud.make_FPFHEstimation()
        est.set_InputNormals(normal_est.compute_cloud())
        est.set_RadiusSearch(0.12)
        return est.compute()

    model_desc = describe(model)
    scene_desc = describe(scene)

    pairs = pcl.match_descriptors(model_desc, scene_desc)
    assert len(pairs) == scene.size

    grouping = pcl.GeometricConsistencyGrouping(model, scene)
    grouping.set_GCSize(0.05)
    grouping.set_GCThreshold(10)
    found = grouping.recognize(pairs)
    assert found

    # Ambiguous matches on flat regions can seed small spurious
    # clusters; the best-supported instance is the object.
    transform, correspondences = max(found, key=lambda f: len(f[1]))
    assert len(correspondences) >= 50
    assert transform[:3, 3] == pytest.approx(translation, abs=0.05)
    assert transform[:3, :3] == pytest.approx(np.eye(3), abs=0.05)


# --- descriptor matching -----------------------------------------------

def test_match_agrees_with_brute_force():
    """FLANN must return exactly the nearest neighbours numpy finds."""
    rng = np.random.RandomState(0)
    model = rng.rand(200, 33).astype(np.float32) * 10
    scene = model[rng.permutation(200)]

    pairs = pcl.match_descriptors(model, scene)
    assert len(pairs) == 200
    squared = ((scene[:, None, :] - model[None, :, :]) ** 2).sum(axis=2)
    expected = squared.argmin(axis=1)
    for model_index, scene_index, distance in pairs:
        assert expected[scene_index] == model_index
        assert distance == pytest.approx(0.0, abs=1e-4)


def test_match_skips_nan_rows_on_both_sides():
    """NaN is SHOT's "no descriptor here"; FLANN would index or match it
    as garbage rather than failing."""
    rng = np.random.RandomState(1)
    model = rng.rand(50, 33).astype(np.float32)
    scene = model.copy()
    model[7] = np.nan
    scene[5] = np.nan

    pairs = pcl.match_descriptors(model, scene)
    assert len(pairs) == 49
    assert all(scene_index != 5 for _, scene_index, _ in pairs)
    assert all(model_index != 7 for model_index, _, _ in pairs)


def test_match_max_distance_bounds_the_matches():
    rng = np.random.RandomState(2)
    model = rng.rand(100, 33).astype(np.float32)
    scene = rng.rand(100, 33).astype(np.float32)

    bounded = pcl.match_descriptors(model, scene, max_distance=0.5)
    unbounded = pcl.match_descriptors(model, scene)
    assert len(bounded) < len(unbounded) == 100
    assert all(distance <= 0.5 for _, _, distance in bounded)


def test_match_works_for_shot_width_too(box, box_normals):
    descriptors = _shot(box, box_normals).compute()
    pairs = pcl.match_descriptors(descriptors, descriptors)
    assert len(pairs) == box.size
    # Matching a set against itself: everything at distance ~0.
    assert all(distance == pytest.approx(0.0, abs=1e-5)
               for _, _, distance in pairs)


def test_match_rejects_mismatched_and_unknown_widths():
    fpfh_like = np.zeros((5, 33), dtype=np.float32)
    with pytest.raises(ValueError, match="differ in width"):
        pcl.match_descriptors(fpfh_like, np.zeros((5, 352), dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported descriptor width"):
        pcl.match_descriptors(np.zeros((5, 40), dtype=np.float32),
                              np.zeros((5, 40), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        pcl.match_descriptors(np.zeros(33, dtype=np.float32), fpfh_like)


def test_match_of_empty_input_is_empty():
    empty = np.zeros((0, 33), dtype=np.float32)
    some = np.zeros((4, 33), dtype=np.float32)
    assert pcl.match_descriptors(empty, some) == []
    assert pcl.match_descriptors(some, empty) == []
