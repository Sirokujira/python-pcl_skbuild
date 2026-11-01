"""Runtime tests for the third registration batch: more transformation
estimators, correspondence estimators and rejectors, GICP6D, NDT2D,
joint ICP, PPF registration, incremental / meta registration, LUM, ELCH
and the pyramid feature match.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _matrix(rotation, translation, scale=1.0):
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = scale * rotation
    out[:3, 3] = translation
    return out


def _apply(m, pts):
    return (pts @ m[:3, :3].T + m[:3, 3]).astype(np.float32)


@pytest.fixture
def blob():
    return np.random.RandomState(0).rand(300, 3).astype(np.float32)


@pytest.fixture
def surface():
    """A wavy height field with analytic normals, as PointNormal rows."""
    u, v = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    u, v = u.ravel(), v.ravel()
    z = 0.1 * np.sin(4 * u) * np.cos(3 * v)
    dzdu = 0.4 * np.cos(4 * u) * np.cos(3 * v)
    dzdv = -0.3 * np.sin(4 * u) * np.sin(3 * v)
    n = np.column_stack([-dzdu, -dzdv, np.ones_like(u)])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return np.column_stack([u, v, z]).astype(np.float32), n.astype(np.float32)


def _pn(points, normals):
    return pcl.PointCloud_PointNormal(np.hstack([
        points, normals, np.zeros((len(points), 1), np.float32)]).astype(
        np.float32))


# --- transformation estimation -------------------------------------------


@pytest.mark.parametrize("method", ["dual_quaternion", "svd"])
def test_rigid_estimators_recover_a_motion(blob, method):
    truth = _matrix(_rot_z(0.4), [0.3, -0.2, 0.1])
    est = pcl.estimate_transformation(pcl.PointCloud(blob),
                                      pcl.PointCloud(_apply(truth, blob)),
                                      method=method)
    assert est == pytest.approx(truth, abs=1e-4)


def test_svd_scale_recovers_the_scale(blob):
    truth = _matrix(_rot_z(0.3), [0.1, 0.2, 0.3], scale=1.7)
    est = pcl.estimate_transformation(pcl.PointCloud(blob),
                                      pcl.PointCloud(_apply(truth, blob)),
                                      method="svd_scale")
    assert est == pytest.approx(truth, abs=1e-3)


def test_2d_estimator_ignores_out_of_plane_motion(blob):
    truth = _matrix(_rot_z(0.5), [0.3, -0.4, 0.0])
    est = pcl.estimate_transformation(pcl.PointCloud(blob),
                                      pcl.PointCloud(_apply(truth, blob)),
                                      method="2d")
    assert est == pytest.approx(truth, abs=1e-3)


def test_three_point_estimator(blob):
    truth = _matrix(_rot_z(0.2), [0.5, 0.0, 0.0])
    src = pcl.PointCloud(blob)
    tgt = pcl.PointCloud(_apply(truth, blob))
    pairs = np.array([[0, 0], [10, 10], [20, 20]])
    est = pcl.estimate_transformation(src, tgt, pairs, method="3point")
    assert est == pytest.approx(truth, abs=1e-3)
    with pytest.raises(ValueError, match="three"):
        pcl.estimate_transformation(src, tgt, method="3point")


@pytest.mark.parametrize("method", ["lls", "lm", "symmetric", "lls_weighted",
                                    "weighted"])
def test_point_to_plane_estimators(surface, method):
    pts, normals = surface
    truth = _matrix(_rot_z(0.03), [0.01, -0.01, 0.005])
    moved_n = normals @ truth[:3, :3].T
    src = _pn(pts, normals)
    tgt = _pn(_apply(truth, pts), moved_n.astype(np.float32))
    weights = None
    if "weighted" in method:
        weights = np.ones(len(pts))
    est = pcl.estimate_transformation_point_to_plane(src, tgt, method=method,
                                                     weights=weights)
    assert est == pytest.approx(truth, abs=2e-3)


def test_point_to_plane_checks():
    tiny = _pn(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError):
        pcl.estimate_transformation_point_to_plane(tiny, tiny)
    pts = np.random.rand(10, 3).astype(np.float32)
    cloud = _pn(pts, np.tile([0, 0, 1], (10, 1)).astype(np.float32))
    with pytest.raises(ValueError, match="weights"):
        pcl.estimate_transformation_point_to_plane(cloud, cloud,
                                                   method="weighted")
    with pytest.raises(ValueError):
        pcl.estimate_transformation_point_to_plane(cloud, cloud,
                                                   method="nope")


# --- correspondence estimation / rejection --------------------------------


def _normal_cloud(normals):
    v = np.zeros((len(normals), 4), np.float32)
    v[:, :3] = normals
    return pcl.PointCloud_Normal(v)


def test_normal_shooting_and_back_projection(surface):
    pts, normals = surface
    shifted = (pts + [0.0, 0.0, 0.005]).astype(np.float32)
    src, tgt = pcl.PointCloud(pts), pcl.PointCloud(shifted)
    pairs, d = pcl.estimate_correspondences_normal_shooting(
        src, _normal_cloud(normals), tgt, k=5)
    assert pairs.shape == (len(pts), 2)
    assert np.mean(pairs[:, 0] == pairs[:, 1]) > 0.9
    pairs, d = pcl.estimate_correspondences_back_projection(
        src, _normal_cloud(normals), tgt, _normal_cloud(normals), k=5,
        max_distance=0.05)
    assert len(pairs) > 0.9 * len(pts)
    assert np.all(d <= 0.05 ** 2 + 1e-7)
    with pytest.raises(ValueError):
        pcl.estimate_correspondences_normal_shooting(
            src, _normal_cloud(normals[:3]), tgt)


def _organized(depth, f=50.0):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    # pixel centres: PCL truncates the projected coordinate
    u += 0.5
    v += 0.5
    xyz = np.dstack([(u - w / 2) * depth / f, (v - h / 2) * depth / f, depth])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    return cloud


def test_organized_projection_matches_pixels():
    depth = np.full((30, 40), 2.0, np.float32)
    target = _organized(depth)
    source = pcl.PointCloud(target.to_array()[::7])
    pairs, d = pcl.estimate_correspondences_organized_projection(
        source, target, 50.0, 50.0, 20.0, 15.0, depth_threshold=0.1)
    assert len(pairs) > 0.9 * source.size
    assert np.all(pairs[:, 1] == pairs[:, 0] * 7)
    with pytest.raises(ValueError):
        pcl.estimate_correspondences_organized_projection(
            source, source, 50.0, 50.0, 20.0, 15.0)


def test_var_trimmed_drops_the_outliers(blob):
    src = pcl.PointCloud(blob)
    moved = blob.copy()
    moved[:60] += 0.5
    tgt = pcl.PointCloud(moved)
    pairs = np.column_stack([np.arange(300), np.arange(300)])
    d = np.sum((blob - moved) ** 2, axis=1).astype(np.float32)
    kept, kd, trimmed, factor = pcl.reject_var_trimmed(src, tgt, pairs, d)
    assert len(kept) < 300
    assert 0.05 <= factor <= 0.95
    assert not np.isin(np.arange(60), kept[:, 0]).all()


def test_poly_rejector(blob):
    truth = _matrix(_rot_z(0.3), [0.2, 0.1, 0.0])
    src = pcl.PointCloud(blob)
    tgt = pcl.PointCloud(_apply(truth, blob))
    pairs = np.column_stack([np.arange(300), np.arange(300)])
    rng = np.random.RandomState(1)
    bad = rng.choice(300, 60, replace=False)
    pairs[bad, 1] = rng.randint(0, 300, 60)
    d = np.zeros(300, np.float32)
    kept, _ = pcl.reject_poly(src, tgt, pairs, d, iterations=20000)
    good = set(np.setdiff1d(np.arange(300), bad).tolist())
    assert len(set(kept[:, 0].tolist()) & good) > 0.8 * len(good)
    assert len(set(kept[:, 0].tolist()) & set(bad.tolist())) < 15


def test_organized_boundary_rejects_edge_matches():
    depth = np.full((30, 40), 2.0, np.float32)
    depth[:, 20:] = np.nan
    target = _organized(depth)
    pairs = np.column_stack([np.arange(target.size), np.arange(target.size)])
    finite = np.isfinite(target.to_array()[:, 2])
    pairs = pairs[finite]
    d = np.zeros(len(pairs), np.float32)
    kept, _ = pcl.reject_organized_boundary(target, target, pairs, d)
    assert 0 < len(kept) < len(pairs)
    cols = kept[:, 1] % 40
    rows = kept[:, 1] // 40
    # column 19 sees two invalid columns (10 pixels) in its 5 x 5 window,
    # column 18 one (5); rows 0 and 29 have a clipped 3-row window
    interior = (rows > 0) & (rows < 29)
    assert cols[interior].max() == 18
    assert set(rows[cols == 19]) == {0, 29}


# --- align-style registrations ------------------------------------------------


def test_gicp6d(blob):
    truth = _matrix(_rot_z(0.05), [0.02, 0.0, 0.0])
    colours = np.zeros((300, 4), np.uint8)
    colours[:, :3] = (blob * 255).astype(np.uint8)
    colours[:, 3] = 255
    src = pcl.PointCloud_PointXYZRGBA()
    src.from_rgba_array(blob, colours)
    tgt = pcl.PointCloud_PointXYZRGBA()
    tgt.from_rgba_array(_apply(truth, blob), colours)
    gicp = pcl.GeneralizedIterativeClosestPoint6D()
    gicp.set_MaxCorrespondenceDistance(0.2)
    converged, m, estimate, fitness = gicp.gicp(src, tgt, max_iter=50)
    assert isinstance(estimate, pcl.PointCloud_PointXYZRGBA)
    assert m[:3, 3] == pytest.approx(truth[:3, 3], abs=0.01)


def test_ndt_2d_recovers_a_planar_motion():
    rng = np.random.RandomState(2)
    pts = np.column_stack([rng.rand(3000) * 8 - 4, rng.rand(3000) * 8 - 4,
                           np.zeros(3000)])
    # a smoothly varying density: NDT needs spread inside each cell
    density = 0.5 + 0.5 * np.sin(pts[:, 0] * 1.3) * np.cos(pts[:, 1] * 0.9)
    pts = pts[rng.rand(len(pts)) < density].astype(np.float32)
    truth = _matrix(_rot_z(0.05), [0.2, -0.1, 0.0])
    ndt = pcl.NormalDistributionsTransform2D()
    ndt.set_Grid((0, 0), (12, 12), (1, 1))
    converged, m, estimate, _ = ndt.ndt(pcl.PointCloud(pts),
                                        pcl.PointCloud(_apply(truth, pts)))
    assert converged
    assert estimate.size == len(pts)
    assert m[:2, 3] == pytest.approx(truth[:2, 3], abs=0.01)
    assert np.arctan2(m[1, 0], m[0, 0]) == pytest.approx(0.05, abs=0.005)


def test_joint_icp_solves_one_motion_for_two_pairs(blob):
    truth = _matrix(_rot_z(0.04), [0.02, 0.01, 0.0])
    a, b = blob[:150], blob[150:] + [2.0, 0, 0]
    icp = pcl.JointIterativeClosestPoint()
    icp.set_MaxCorrespondenceDistance(0.2)
    converged, m, fitness = icp.icp(
        [pcl.PointCloud(a), pcl.PointCloud(b.astype(np.float32))],
        [pcl.PointCloud(_apply(truth, a)),
         pcl.PointCloud(_apply(truth, b.astype(np.float32)))], max_iter=50)
    assert m == pytest.approx(truth, abs=0.01)
    with pytest.raises(ValueError):
        icp.icp([pcl.PointCloud(a)], [])


def test_ppf_registration_finds_the_model(surface):
    pts, normals = surface
    model = _pn(pts[::3], normals[::3])
    truth = _matrix(_rot_z(0.6), [0.5, 0.2, 0.1])
    scene = _pn(_apply(truth, pts[::3]), (normals[::3] @ truth[:3, :3].T)
                .astype(np.float32))
    ppf = pcl.PPFRegistration(angle_step=np.radians(12), distance_step=0.05)
    ppf.set_SceneReferencePointSamplingRate(5)
    converged, m, estimate, _ = ppf.align(model, scene)
    assert isinstance(estimate, pcl.PointCloud_PointNormal)
    moved = _apply(m, pts[::3])
    assert np.median(np.linalg.norm(moved - _apply(truth, pts[::3]),
                                    axis=1)) < 0.1


# --- frameworks --------------------------------------------------------------


def _frames(blob, steps=4):
    step = _matrix(_rot_z(0.02), [0.01, 0.0, 0.0])
    frames, current = [], blob
    for _ in range(steps):
        frames.append(pcl.PointCloud(current))
        current = _apply(step, current)
    return frames, step


def test_incremental_registration_composes_the_steps(blob):
    frames, step = _frames(blob)
    reg = pcl.IncrementalRegistration(max_correspondence_distance=0.1)
    for f in frames:
        assert reg.register_Cloud(f)
    # Frame k+1 = step * frame k, so aligning back accumulates step^-1.
    expected = np.linalg.matrix_power(np.linalg.inv(step), len(frames) - 1)
    assert reg.get_AbsoluteTransform() == pytest.approx(expected, abs=0.01)
    assert reg.get_DeltaTransform() == pytest.approx(np.linalg.inv(step),
                                                     abs=0.01)
    reg.reset()


def test_meta_registration_builds_a_map(blob):
    frames, _ = _frames(blob, steps=3)
    reg = pcl.MetaRegistration(max_correspondence_distance=0.1)
    for f in frames:
        reg.register_Cloud(f)
    assert reg.get_MetaCloud().size == 3 * len(blob)
    assert reg.get_AbsoluteTransform().shape == (4, 4)


def test_lum_moves_poses_to_agree(blob):
    a = pcl.PointCloud(blob)
    b = pcl.PointCloud(blob)                 # same points, same frame...
    lum = pcl.LUM()
    va = lum.add_PointCloud(a)
    vb = lum.add_PointCloud(b, pose=(0.05, 0, 0, 0, 0, 0))   # ...offset pose
    pairs = np.column_stack([np.arange(300), np.arange(300)])
    lum.set_Correspondences(vb, va, pairs)
    lum.set_MaxIterations(20)
    lum.compute()
    assert lum.get_NumVertices() == 2
    assert np.abs(lum.get_Pose(vb)[:3]).max() < 0.01
    assert lum.get_Transformation(vb).shape == (4, 4)
    assert lum.get_ConcatenatedCloud().size == 600
    with pytest.raises(IndexError):
        lum.get_Pose(5)


def test_elch_closes_a_loop(blob):
    chain = [pcl.PointCloud((blob + [i * 0.5, 0, 0]).astype(np.float32))
             for i in range(4)]
    chain.append(pcl.PointCloud((blob + [0.02, 0.0, 0.0]).astype(np.float32)))
    elch = pcl.ELCH(max_correspondence_distance=0.1)
    for c in chain:
        elch.add_PointCloud(c)
    elch.set_LoopStart(0)
    elch.set_LoopEnd(4)
    elch.compute()
    end = elch.get_PointCloud(4).to_array()
    assert np.abs(end - blob).mean() < 0.02
    with pytest.raises(RuntimeError):
        pcl.ELCH().compute()


# --- pyramid match --------------------------------------------------------------


def test_pyramid_match_prefers_similar_sets():
    rng = np.random.RandomState(3)
    a = rng.rand(300, 4).astype(np.float32)
    b = np.clip(a + rng.randn(*a.shape).astype(np.float32) * 0.01, 0, 0.999)
    c = (rng.rand(300, 4) ** 4).astype(np.float32)
    ranges = np.tile([0.0, 1.0], (4, 1))
    same = pcl.compare_pyramid_feature_histograms(a, b, ranges)
    other = pcl.compare_pyramid_feature_histograms(a, c, ranges)
    assert pcl.compare_pyramid_feature_histograms(a, a, ranges) == \
        pytest.approx(1.0)
    assert 0.0 <= other < same <= 1.0 + 1e-6
    # the default range spans both sets
    assert 0.0 <= pcl.compare_pyramid_feature_histograms(a, c) <= 1.0 + 1e-6


def test_pyramid_match_refuses_what_pcl_cannot_index():
    rng = np.random.RandomState(4)
    a = rng.rand(50, 4).astype(np.float32)
    ranges = np.tile([0.0, 1.0], (4, 1))
    with pytest.raises(ValueError, match="dense"):
        pcl.compare_pyramid_feature_histograms(rng.rand(10, 33),
                                               rng.rand(10, 33))
    with pytest.raises(ValueError, match="same d"):
        pcl.compare_pyramid_feature_histograms(a, a[:, :3])
    with pytest.raises(ValueError, match="past the end"):
        pcl.compare_pyramid_feature_histograms(a, a + 1.0, ranges)
    with pytest.raises(ValueError, match="limit"):
        pcl.compare_pyramid_feature_histograms(
            a, a, ranges, np.tile([0.0, 1000.0], (4, 1)))


# --- SAC models ---------------------------------------------------------------


def test_ellipse_3d_model():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    pts = np.column_stack([1 + 0.5 * np.cos(t), 2 + 0.2 * np.sin(t),
                           np.full_like(t, 3.0)]).astype(np.float32)
    model = pcl.SampleConsensusModelEllipse3D(pcl.PointCloud(pts))
    sac = pcl.RandomSampleConsensus(model)
    sac.set_DistanceThreshold(0.01)
    assert sac.computeModel()
    coefficients = sac.get_ModelCoefficients()
    assert len(coefficients) == 11
    assert coefficients[:3] == pytest.approx([1, 2, 3], abs=0.02)
    assert sorted(coefficients[3:5]) == pytest.approx([0.2, 0.5], abs=0.02)


@pytest.mark.parametrize("name", ["SACMODEL_TORUS", "SACMODEL_PARALLEL_LINES",
                                  "SACMODEL_REGISTRATION_2D"])
def test_unbuildable_models_say_why(name):
    with pytest.raises(ValueError):
        pcl.SampleConsensusModel(pcl.PointCloud(np.zeros((5, 3), np.float32)),
                                 getattr(pcl, name))
