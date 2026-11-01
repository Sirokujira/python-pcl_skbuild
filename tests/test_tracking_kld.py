"""Runtime tests for KLDAdaptiveParticleFilterTracker.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


BLOCK = np.mgrid[0:8, 0:8, 0:3].reshape(3, -1).T * 0.05


def _moved(offset):
    return pcl.PointCloud((BLOCK + offset).astype(np.float32))


def _track(offsets, **kwargs):
    kwargs.setdefault("threads", 2)
    tracker = pcl.KLDAdaptiveParticleFilterTracker(**kwargs)
    tracker.set_ReferenceCloud(_moved(np.zeros(3)))
    poses = []
    for offset in offsets:
        tracker.set_InputCloud(_moved(offset))
        tracker.compute()
        poses.append(tracker.get_Result())
    return tracker, poses


def test_kld_tracker_follows_a_moving_object():
    offsets = [np.array([0.01 * f, 0.0, 0.0]) for f in range(1, 9)]
    _, poses = _track(offsets)
    assert poses[-1][0] - poses[0][0] > 0.02


def test_kld_tracker_is_a_particle_filter_tracker():
    tracker, poses = _track([np.zeros(3)] * 3)
    assert isinstance(tracker, pcl.ParticleFilterTracker)
    assert len(poses[-1]) == 7
    assert tracker.get_ResultTransform().shape == (4, 4)
    assert tracker.get_AlignedReference().size == len(BLOCK)


def test_kld_tracker_adapts_the_particle_count():
    tracker, _ = _track([np.zeros(3)] * 4, particle_num=100,
                        max_particle_num=800)
    assert 1 <= tracker.particle_num <= 800


def test_kld_tracker_checks_its_settings():
    with pytest.raises(ValueError):
        pcl.KLDAdaptiveParticleFilterTracker(particle_num=500,
                                             max_particle_num=100)
    with pytest.raises(ValueError):
        pcl.KLDAdaptiveParticleFilterTracker(bin_size=0)
    with pytest.raises(ValueError):
        pcl.KLDAdaptiveParticleFilterTracker(delta=1.0)
    with pytest.raises(ValueError):
        pcl.KLDAdaptiveParticleFilterTracker(epsilon=0)
    with pytest.raises(RuntimeError):
        pcl.KLDAdaptiveParticleFilterTracker().compute()
