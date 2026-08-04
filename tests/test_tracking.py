"""Runtime tests for particle-filter tracking (pcl/tracking).

A particle filter is stochastic, so these check that it FOLLOWS — the
tracked pose moves with the object and ends up near it — rather than
asserting an exact pose it has no obligation to reach.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


BLOCK = np.mgrid[0:8, 0:8, 0:3].reshape(3, -1).T * 0.05


@pytest.fixture
def model():
    return pcl.PointCloud(BLOCK.astype(np.float32))


def _moved(offset):
    return pcl.PointCloud((BLOCK + offset).astype(np.float32))


def _tracker(model, **kwargs):
    kwargs.setdefault("particle_num", 600)
    kwargs.setdefault("step_noise", 0.015)
    kwargs.setdefault("threads", 2)
    tracker = pcl.ParticleFilterTracker(**kwargs)
    tracker.set_ReferenceCloud(model)
    return tracker


def _track(model, offsets, **kwargs):
    tracker = _tracker(model, **kwargs)
    poses = []
    for offset in offsets:
        tracker.set_InputCloud(_moved(offset))
        tracker.compute()
        poses.append(tracker.get_Result())
    return tracker, poses


# --- following ---------------------------------------------------------

def test_tracker_follows_a_moving_object(model):
    """The reported x must travel with the object, not sit still."""
    offsets = [np.array([0.01 * f, 0.0, 0.0]) for f in range(1, 9)]
    _, poses = _track(model, offsets)
    travelled = poses[-1][0] - poses[0][0]
    assert travelled > 0.02


def test_tracker_stays_put_on_a_still_object(model):
    """The complement of the test above: without motion the pose must
    not wander off."""
    offsets = [np.zeros(3)] * 8
    _, poses = _track(model, offsets)
    assert abs(poses[-1][0] - poses[0][0]) < 0.05


def test_tracked_pose_is_in_scene_coordinates(model):
    """PCL wants the reference in the object's own frame; the wrapper
    recentres it and keeps the centroid, so a caller never has to."""
    centroid = BLOCK.mean(axis=0)
    _, poses = _track(model, [np.zeros(3)] * 4)
    x, y, z = poses[-1][:3]
    assert np.allclose([x, y, z], centroid, atol=0.1)


def test_result_has_the_full_particle_state(model):
    _, poses = _track(model, [np.zeros(3)])
    assert len(poses[-1]) == 7
    assert all(isinstance(value, float) for value in poses[-1])
    assert np.isfinite(poses[-1]).all()


def test_result_weight_is_a_probability(model):
    _, poses = _track(model, [np.zeros(3)] * 3)
    weight = poses[-1][6]
    assert 0.0 <= weight <= 1.0


# --- the derived forms -------------------------------------------------

def test_result_transform_matches_the_state(model):
    tracker, poses = _track(model, [np.array([0.02, 0.0, 0.0])] * 4)
    transform = tracker.get_ResultTransform()
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float32
    assert transform[:3, 3] == pytest.approx(poses[-1][:3], abs=1e-4)
    assert transform[3] == pytest.approx([0, 0, 0, 1])


def test_aligned_reference_is_the_model_at_the_tracked_pose(model):
    tracker, poses = _track(model, [np.array([0.02, 0.0, 0.0])] * 4)
    aligned = tracker.get_AlignedReference()
    assert isinstance(aligned, pcl.PointCloud)
    assert aligned.size == model.size
    # Its centroid is where the tracker says the object is.
    assert aligned.to_array().mean(axis=0) == pytest.approx(
        np.asarray(poses[-1][:3]), abs=1e-3)


# --- construction and error paths --------------------------------------

def test_particle_num_is_what_was_asked_for(model):
    assert _tracker(model, particle_num=250).particle_num == 250


def test_tracker_rejects_nonsense_settings():
    with pytest.raises(ValueError, match="particle_num"):
        pcl.ParticleFilterTracker(particle_num=0)
    with pytest.raises(ValueError, match="step_noise"):
        pcl.ParticleFilterTracker(step_noise=0.0)
    with pytest.raises(ValueError, match="resolution"):
        pcl.ParticleFilterTracker(resolution=0.0)
    with pytest.raises(ValueError, match="threads"):
        pcl.ParticleFilterTracker(threads=0)


def test_tracker_rejects_an_empty_reference():
    tracker = pcl.ParticleFilterTracker()
    with pytest.raises(ValueError, match="empty"):
        tracker.set_ReferenceCloud(pcl.PointCloud())


def test_compute_without_a_reference_reports_it(model):
    tracker = pcl.ParticleFilterTracker()
    tracker.set_InputCloud(model)
    with pytest.raises(RuntimeError, match="set_ReferenceCloud"):
        tracker.compute()


def test_compute_without_an_input_reports_it(model):
    tracker = pcl.ParticleFilterTracker()
    tracker.set_ReferenceCloud(model)
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        tracker.compute()
