"""Runtime tests for correspondence grouping (pcl/recognition).

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


TRANSLATION = np.array([1.0, 2.0, 3.0])


@pytest.fixture
def model():
    """A solid block of points — enough structure for a unique pose."""
    grid = np.mgrid[0:8, 0:8, 0:3].reshape(3, -1).T * 0.05
    return pcl.PointCloud(grid.astype(np.float32))


@pytest.fixture
def scene(model):
    """The same block, moved by a known amount."""
    return pcl.PointCloud((model.to_array() + TRANSLATION).astype(np.float32))


@pytest.fixture
def perfect_pairs(model):
    """Index-for-index correspondences, as a perfect matcher would give."""
    return [(i, i) for i in range(model.size)]


def _gc(model, scene):
    grouping = pcl.GeometricConsistencyGrouping(model, scene)
    grouping.set_GCSize(0.05)
    grouping.set_GCThreshold(5)
    return grouping


def _hough(model, scene):
    grouping = pcl.Hough3DGrouping(model, scene)
    grouping.set_RFRadius(0.1)
    grouping.set_HoughBinSize(0.05)
    grouping.set_HoughThreshold(5.0)
    return grouping


# --- geometric consistency -------------------------------------------

def test_gc_finds_the_model(model, scene, perfect_pairs):
    assert len(_gc(model, scene).recognize(perfect_pairs)) == 1


def test_gc_recovers_the_transform(model, scene, perfect_pairs):
    transform, _ = _gc(model, scene).recognize(perfect_pairs)[0]
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float32
    assert transform[:3, 3] == pytest.approx(TRANSLATION, abs=1e-3)
    # No rotation was applied, so the upper-left block is the identity.
    assert transform[:3, :3] == pytest.approx(np.eye(3), abs=1e-3)
    assert transform[3] == pytest.approx([0, 0, 0, 1])


def test_gc_returns_the_supporting_correspondences(model, scene,
                                                   perfect_pairs):
    _, correspondences = _gc(model, scene).recognize(perfect_pairs)[0]
    assert len(correspondences) >= 5
    for model_index, scene_index in correspondences:
        assert 0 <= model_index < model.size
        assert 0 <= scene_index < scene.size


def test_gc_accepts_distances_on_the_correspondences(model, scene):
    """A real matcher carries a descriptor distance; it must be optional,
    not forbidden."""
    with_distance = [(i, i, 0.25) for i in range(model.size)]
    assert len(_gc(model, scene).recognize(with_distance)) == 1


def test_gc_rejects_a_malformed_correspondence(model, scene):
    with pytest.raises(ValueError, match="model_index"):
        _gc(model, scene).recognize([(0, 0, 0.0, 99)])


def test_gc_finds_nothing_without_correspondences(model, scene):
    assert _gc(model, scene).recognize([]) == []


def test_gc_finds_nothing_in_random_correspondences(model, scene):
    """Matches that no rigid transform explains must not become a pose —
    that discarding is the whole point of grouping."""
    rng = np.random.RandomState(0)
    noise = [(int(i), int(j)) for i, j in
             zip(rng.randint(0, model.size, 60),
                 rng.randint(0, scene.size, 60))]
    grouping = pcl.GeometricConsistencyGrouping(model, scene)
    grouping.set_GCSize(0.01)
    grouping.set_GCThreshold(20)
    assert grouping.recognize(noise) == []


def test_gc_settings_roundtrip():
    grouping = pcl.GeometricConsistencyGrouping()
    grouping.set_GCSize(0.02)
    grouping.set_GCThreshold(7)
    assert grouping.get_GCSize() == pytest.approx(0.02)
    assert grouping.get_GCThreshold() == 7


def test_gc_rejects_a_threshold_below_three():
    """Three correspondences is PCL's floor: a 6-DOF pose needs them."""
    with pytest.raises(ValueError, match="6-DOF"):
        pcl.GeometricConsistencyGrouping().set_GCThreshold(2)


def test_gc_rejects_a_nonpositive_size():
    with pytest.raises(ValueError, match="gc_size"):
        pcl.GeometricConsistencyGrouping().set_GCSize(0.0)


def test_gc_without_clouds_reports_it(model, perfect_pairs):
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.GeometricConsistencyGrouping().recognize(perfect_pairs)

    grouping = pcl.GeometricConsistencyGrouping()
    grouping.set_InputCloud(model)
    with pytest.raises(RuntimeError, match="set_SceneCloud"):
        grouping.recognize(perfect_pairs)


# --- Hough voting ------------------------------------------------------

def test_hough_recovers_the_transform(model, scene, perfect_pairs):
    found = _hough(model, scene).recognize(perfect_pairs)
    assert len(found) == 1
    transform, correspondences = found[0]
    assert transform[:3, 3] == pytest.approx(TRANSLATION, abs=1e-2)
    assert len(correspondences) >= 3


def test_hough_settings_roundtrip():
    grouping = pcl.Hough3DGrouping()
    grouping.set_RFRadius(0.05)
    grouping.set_HoughBinSize(0.02)
    grouping.set_HoughThreshold(9.0)
    assert grouping.get_RFRadius() == pytest.approx(0.05)
    assert grouping.get_HoughBinSize() == pytest.approx(0.02)
    assert grouping.get_HoughThreshold() == pytest.approx(9.0)


def test_hough_rejects_nonpositive_settings():
    grouping = pcl.Hough3DGrouping()
    with pytest.raises(ValueError, match="rf_radius"):
        grouping.set_RFRadius(0.0)
    with pytest.raises(ValueError, match="bin_size"):
        grouping.set_HoughBinSize(0.0)


def test_hough_without_clouds_reports_it(perfect_pairs):
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.Hough3DGrouping().recognize(perfect_pairs)


def test_both_algorithms_agree_on_the_pose(model, scene, perfect_pairs):
    gc_transform, _ = _gc(model, scene).recognize(perfect_pairs)[0]
    hough_transform, _ = _hough(model, scene).recognize(perfect_pairs)[0]
    assert gc_transform[:3, 3] == pytest.approx(hough_transform[:3, 3],
                                                abs=1e-2)
