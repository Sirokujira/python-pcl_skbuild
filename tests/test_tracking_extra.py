"""Particle-filter coherences beyond distance (colour, normals, exact
pairing) and PyramidalKLTTracker.

The particle filter is stochastic, so its tests check that it FOLLOWS,
as in test_tracking.py. KLT is deterministic and is checked against the
shift it was given."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._tracking")


BLOCK = (np.mgrid[0:8, 0:8, 0:3].reshape(3, -1).T * 0.05).astype(np.float32)
OFFSETS = [np.array([0.01 * f, 0.0, 0.0], np.float32) for f in range(1, 9)]


def _rgba(xyz, colour=(220, 30, 30)):
    rgba = np.zeros((len(xyz), 4), np.uint8)
    rgba[:, :3] = colour
    rgba[:, 3] = 255
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(np.ascontiguousarray(xyz, np.float32), rgba)
    return cloud


def _box_surface(side=0.3, step=0.05):
    """Points on the six faces of a cube 1 m in front of the origin, each
    with its face's normal (PCL keeps only the faces turned towards the
    origin when it scores normals)."""
    ticks = np.arange(0.0, side + 1e-6, step, dtype=np.float32)
    a, b = np.meshgrid(ticks, ticks)
    a, b = a.ravel(), b.ravel()
    rows = []
    for axis in range(3):
        for level, sign in ((0.0, -1.0), (side, 1.0)):
            pts = np.zeros((len(a), 3), np.float32)
            others = [k for k in range(3) if k != axis]
            pts[:, axis] = level
            pts[:, others[0]] = a
            pts[:, others[1]] = b
            normals = np.zeros_like(pts)
            normals[:, axis] = sign
            rows.append(np.hstack([pts, normals,
                                   np.zeros((len(a), 1), np.float32)]))
    surface = np.vstack(rows)
    surface[:, :3] += (-side / 2, -side / 2, 1.0)
    return surface


def _follow(tracker, clouds):
    poses = []
    for cloud in clouds:
        tracker.set_InputCloud(cloud)
        tracker.compute()
        poses.append(tracker.get_Result())
    return poses


def test_colour_coherence_follows_a_coloured_object():
    tracker = pcl.ParticleFilterTracker(particle_num=600, threads=2)
    tracker.add_HSVColorCoherence()
    tracker.set_ReferenceCloud(_rgba(BLOCK))
    poses = _follow(tracker, [_rgba(BLOCK + o) for o in OFFSETS])
    assert poses[-1][0] - poses[0][0] > 0.02
    aligned = tracker.get_AlignedReference()
    assert isinstance(aligned, pcl.PointCloud_PointXYZRGBA)
    assert aligned.size == len(BLOCK)


def test_normal_coherence_follows_a_box():
    """PCL's OMP trackers drop every update once normals are on; the
    shim restores them (see WeighingNormals in compat/tracking_args.h)."""
    surface = _box_surface()
    tracker = pcl.ParticleFilterTracker(particle_num=600, threads=2)
    tracker.add_NormalCoherence()
    tracker.set_ReferenceCloud(pcl.PointCloud_PointNormal(surface))
    moved = []
    for offset in OFFSETS:
        frame = surface.copy()
        frame[:, :3] += offset
        moved.append(pcl.PointCloud_PointNormal(frame))
    poses = _follow(tracker, moved)
    assert poses[-1][0] - poses[0][0] > 0.02
    assert isinstance(tracker.get_AlignedReference(),
                      pcl.PointCloud_PointNormal)


def test_exact_coherence_follows():
    tracker = pcl.ParticleFilterTracker(particle_num=600, threads=2,
                                        exact_coherence=True)
    tracker.set_ReferenceCloud(pcl.PointCloud(BLOCK))
    poses = _follow(tracker, [pcl.PointCloud(BLOCK + o) for o in OFFSETS])
    assert poses[-1][0] - poses[0][0] > 0.02


def test_kld_tracker_takes_colour_too():
    tracker = pcl.KLDAdaptiveParticleFilterTracker(particle_num=400,
                                                   max_particle_num=1000,
                                                   threads=2)
    tracker.add_HSVColorCoherence()
    tracker.set_ReferenceCloud(_rgba(BLOCK))
    poses = _follow(tracker, [_rgba(BLOCK + o) for o in OFFSETS])
    assert poses[-1][0] - poses[0][0] > 0.02


def test_point_type_follows_the_reference():
    tracker = pcl.ParticleFilterTracker()
    tracker.add_HSVColorCoherence()
    with pytest.raises(ValueError, match="XYZRGBA"):
        tracker.set_ReferenceCloud(pcl.PointCloud(BLOCK))

    tracker = pcl.ParticleFilterTracker()
    tracker.add_NormalCoherence()
    with pytest.raises(ValueError, match="PointNormal"):
        tracker.set_ReferenceCloud(_rgba(BLOCK))

    tracker = pcl.ParticleFilterTracker()
    tracker.set_ReferenceCloud(_rgba(BLOCK))
    with pytest.raises(RuntimeError, match="before set_ReferenceCloud"):
        tracker.add_HSVColorCoherence()
    with pytest.raises(TypeError, match="PointCloud_PointXYZRGBA"):
        tracker.set_InputCloud(pcl.PointCloud(BLOCK))
    with pytest.raises(TypeError):
        tracker.set_ReferenceCloud(pcl.PointCloud_PointXYZI(
            np.ones((3, 4), np.float32)))


def test_input_before_reference_waits_for_it():
    tracker = pcl.ParticleFilterTracker(particle_num=300)
    tracker.set_InputCloud(_rgba(BLOCK))
    tracker.set_ReferenceCloud(_rgba(BLOCK))
    tracker.compute()
    assert np.isfinite(tracker.get_Result()).all()


# --- PyramidalKLTTracker -----------------------------------------------------------


F = 60.0


def _texture(u, v):
    return (128 + 50 * np.sin(u / 16.0) * np.cos(v / 13.0)
            + 30 * np.sin((u + v) / 9.0) + 20 * np.cos((u - 2 * v) / 5.0))


def _frame(du=0.0, dv=0.0, h=120, w=160):
    """A textured plane at z = 1 seen by a pinhole camera; the texture is
    shifted by (du, dv) pixels, the geometry stays on the pixel grid."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    grey = np.clip(_texture(u - du, v - dv), 0, 255).astype(np.uint8)
    z = np.ones((h, w), np.float32)
    xyz = np.dstack([(u - w / 2) * z / F, (v - h / 2) * z / F, z])
    rgba = np.dstack([grey, grey, grey, np.full_like(grey, 255)])
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz.astype(np.float32), rgba)
    return cloud


def _grid(lo, hi_u, hi_v, step):
    return np.array([(u, v) for v in range(lo, hi_v, step)
                     for u in range(lo, hi_u, step)], np.float32)


def _klt(points, **kwargs):
    klt = pcl.PyramidalKLTTracker(**kwargs)
    klt.set_NumberOfKeypoints(len(points))
    klt.set_InputCloud(_frame())
    klt.set_PointsToTrack(points)
    klt.compute()
    assert not klt.initialized          # the first frame is the reference
    return klt


@pytest.mark.parametrize("levels", [2, 3, 5])
def test_klt_follows_a_shift(levels):
    """PCL's own tracker reports well under 1 pixel here at every depth."""
    points = _grid(20, 140, 100, 10)
    klt = _klt(points, levels=levels, window_width=9, window_height=9)
    klt.set_InputCloud(_frame(2.5, 1.5))
    klt.compute()
    assert klt.initialized
    uv, status = klt.get_TrackedPoints()
    assert np.all(status == 0)
    # whole-pixel shifts come back exact; between pixels, bilinear
    # interpolation of 8-bit texture costs up to ~0.15 px
    error = np.abs(uv - points - (2.5, 1.5))
    assert error.max() < 0.2 and error.mean() < 0.05
    # the 3-D points under the tracked pixels are 2 columns and 1 row on
    motion = klt.get_Result()
    assert motion.shape == (4, 4)
    assert np.allclose(motion[:3, :3], np.eye(3), atol=1e-3)
    assert motion[:3, 3] == pytest.approx((2 / F, 1 / F, 0), abs=1e-3)


def test_klt_follows_frame_to_frame():
    points = _grid(30, 130, 90, 20)
    klt = _klt(points, levels=3, window_width=9, window_height=9)
    for k in (1, 2, 3):
        klt.set_InputCloud(_frame(1.5 * k, 0.0))
        klt.compute()
        uv, status = klt.get_TrackedPoints()
        assert np.all(status == 0)
        assert np.allclose(uv[:, 0] - points[:, 0], 1.5 * k, atol=0.2)


def _half_flat(du=0.0):
    """Textured left half, uniform right half."""
    cloud = _frame(du)
    rgba = cloud.to_rgba_array().reshape(120, 160, 4)
    rgba[:, 80:, :3] = 100
    xyz = cloud.to_xyz_array().reshape(120, 160, 3)
    out = pcl.PointCloud_PointXYZRGBA()
    out.from_organized_rgba_array(np.ascontiguousarray(xyz),
                                  np.ascontiguousarray(rgba))
    return out


def test_klt_drops_flat_windows_and_follows_the_rest():
    points = np.array([[30, 60], [50, 40], [120, 60], [140, 30]],
                      np.float32)
    klt = pcl.PyramidalKLTTracker(levels=2, window_width=7, window_height=7)
    klt.set_InputCloud(_half_flat())
    klt.set_PointsToTrack(points)
    klt.compute()
    klt.set_InputCloud(_half_flat(2.5))
    klt.compute()
    uv, status = klt.get_TrackedPoints()
    assert list(status) == [0, 0, -2, -2]
    assert np.all(np.isnan(uv[2:]))
    assert np.allclose(uv[:2] - points[:2], (2.5, 0.0), atol=0.2)
    # the next step follows the two survivors only
    klt.set_InputCloud(_half_flat(4.5))
    klt.compute()
    uv, status = klt.get_TrackedPoints()
    assert len(status) == 2 and np.all(status == 0)
    assert np.allclose(uv - points[:2], (4.5, 0.0), atol=0.2)


def test_klt_arguments():
    with pytest.raises(ValueError, match="levels"):
        pcl.PyramidalKLTTracker(levels=1)
    with pytest.raises(ValueError, match="odd"):
        pcl.PyramidalKLTTracker(window_width=8)
    klt = pcl.PyramidalKLTTracker()
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        klt.set_PointsToTrack([[1, 1]])
    with pytest.raises(ValueError, match="organized"):
        klt.set_InputCloud(_rgba(BLOCK))
    klt.set_InputCloud(_frame())
    with pytest.raises(ValueError, match="inside"):
        klt.set_PointsToTrack([[160, 10]])
    with pytest.raises(ValueError):
        klt.set_PointsToTrack(np.zeros((0, 2)))
    with pytest.raises(RuntimeError, match="set_PointsToTrack"):
        klt.compute()
