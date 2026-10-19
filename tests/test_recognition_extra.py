"""CRHAlignment, TrimmedICP, occlusion filtering, ObjRecRANSAC, the
implicit shape model, LineRGBD and DOTMOD on synthetic scenes whose
answer is known."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._recognition_extra")


# --- shapes ------------------------------------------------------------------


def _blob_point(th, ph, wobble=0.0):
    r = 0.1 * (1 + 0.3 * np.sin(3 * th) * np.cos(2 * ph) + wobble * np.cos(ph))
    return np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph),
                     0.7 * r * np.cos(th)], axis=-1)


def _blob(n=50, wobble=0.0):
    """A closed, bumpy surface around the origin with outward normals."""
    th, ph = np.meshgrid(np.pi * np.arange(1, n) / n,
                         np.pi * np.arange(2 * n) / n, indexing="ij")
    th, ph, e = th.ravel(), ph.ravel(), 1e-3
    p = _blob_point(th, ph, wobble)
    dt = _blob_point(th + e, ph, wobble) - _blob_point(th - e, ph, wobble)
    dp = _blob_point(th, ph + e, wobble) - _blob_point(th, ph - e, wobble)
    nrm = np.cross(dt, dp)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm[np.sum(nrm * p, axis=1) < 0] *= -1
    return p.astype(np.float32), nrm.astype(np.float32)


def _cylinder():
    ph = 2 * np.pi * np.arange(40) / 40
    z = np.linspace(-0.15, 0.15, 60)
    ph, z = np.meshgrid(ph, z)
    nrm = np.stack([np.cos(ph), np.sin(ph), 0 * ph], axis=-1).reshape(-1, 3)
    return (np.stack([0.05 * np.cos(ph), 0.05 * np.sin(ph), z], axis=-1)
            .reshape(-1, 3).astype(np.float32), nrm.astype(np.float32))


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _normals(nrm):
    return pcl.PointCloud_Normal(
        np.hstack([nrm, np.zeros((len(nrm), 1))]).astype(np.float32))


def _visible(points, normals, rotation, translation):
    """The object placed at a pose, keeping what faces the origin."""
    p = points @ rotation.T + translation
    n = normals @ rotation.T
    keep = np.sum(n * p, axis=1) < 0
    return p[keep].astype(np.float32), n[keep].astype(np.float32)


# --- CRHAlignment -----------------------------------------------------------------


def _crh(points, normals):
    est = pcl.CRHEstimation(pcl.PointCloud(points))
    est.set_InputNormals(_normals(normals))
    return est.compute()


@pytest.mark.parametrize("roll", [40.0, -60.0, 150.0])
def test_crh_alignment_recovers_the_roll(roll):
    pts, nrm = _blob(60, wobble=0.2)
    rot = _rotation((1, 0.5, 0), 0.4)
    a, an = _visible(pts, nrm, rot, np.array([0.1, -0.05, 1.0]))
    centroid = a.mean(axis=0)
    spin = _rotation(centroid, np.radians(roll))
    b, bn = (a @ spin.T).astype(np.float32), (an @ spin.T).astype(np.float32)
    align = pcl.CRHAlignment()
    candidates = align.align(_crh(b, bn), _crh(a, an), b.mean(axis=0),
                             centroid)
    assert 1 <= len(candidates) <= 5
    # the best one takes the rolled view back onto the original
    best = min(np.abs(b @ m[:3, :3].T + m[:3, 3] - a).mean()
               for m in candidates)
    assert best < 0.005
    with pytest.raises(ValueError):
        align.align(np.zeros(10), np.zeros(90), centroid, centroid)


# --- TrimmedICP -----------------------------------------------------------------------


def test_trimmed_icp_ignores_what_does_not_overlap():
    pts, _ = _blob(40)
    target = pcl.PointCloud(pts)
    motion = np.eye(4, dtype=np.float32)
    motion[:3, :3] = _rotation((0, 0, 1), 0.05)
    motion[:3, 3] = (0.01, -0.005, 0.008)
    # the source: part of the target moved back by the inverse motion, plus
    # a slab of points the target does not have
    inv = np.linalg.inv(motion)
    part = pts[pts[:, 0] > -0.02]
    source = np.vstack([part @ inv[:3, :3].T + inv[:3, 3],
                        np.random.RandomState(1).uniform(0.3, 0.4, (200, 3))])
    icp = pcl.TrimmedICP()
    icp.init(target)
    found = icp.align(pcl.PointCloud(source.astype(np.float32)),
                      int(len(part) * 0.8))
    assert np.allclose(found, motion, atol=2e-3)
    with pytest.raises(ValueError):
        icp.align(pcl.PointCloud(source.astype(np.float32)), len(source))
    with pytest.raises(RuntimeError):
        pcl.TrimmedICP().align(target, 10)


# --- occlusion ---------------------------------------------------------------------------


def _wall(h=30, w=40, f=40.0, z=1.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    xyz = np.dstack([(u - cx) * z / f, (v - cy) * z / f, np.full((h, w), z)])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    return cloud


def test_occlusion_filter_keeps_what_the_wall_does_not_hide():
    wall = _wall()
    front = np.array([[0.0, 0.0, 0.8], [0.1, -0.1, 0.5]], np.float32)
    behind = np.array([[0.05, 0.05, 1.3], [-0.2, 0.1, 2.0]], np.float32)
    outside = np.array([[5.0, 0.0, 1.0]], np.float32)
    cloud = pcl.PointCloud(np.vstack([front, behind, outside]))
    visible = pcl.occlusion_filter(wall, cloud, 40.0, threshold=0.01)
    assert np.allclose(np.sort(visible.to_array(), 0), np.sort(front, 0))
    hidden = pcl.get_occluded_cloud(wall, cloud, 40.0)
    assert np.allclose(np.sort(hidden.to_array(), 0), np.sort(behind, 0))
    # a point just behind the wall passes within the threshold
    near = pcl.PointCloud(np.array([[0.0, 0.0, 1.005]], np.float32))
    assert pcl.occlusion_filter(wall, near, 40.0, threshold=0.01).size == 1
    with pytest.raises(ValueError, match="organized"):
        pcl.occlusion_filter(pcl.PointCloud(front), cloud, 40.0)


# --- ObjRecRANSAC ---------------------------------------------------------------------------


def test_obj_rec_ransac_finds_the_blob():
    pts, nrm = _blob(80)
    rot = _rotation((0.3, 1.0, 0.2), 0.8)
    where = np.array([0.2, -0.1, 1.0])
    seen, seen_n = _visible(pts, nrm, rot, where)
    gx, gz = np.meshgrid(np.arange(-0.4, 0.8, 0.01), np.arange(0.6, 1.6, 0.01))
    floor = np.stack([gx.ravel(), np.full(gx.size, 0.15), gz.ravel()], 1)
    scene = np.vstack([seen, floor]).astype(np.float32)
    scene_n = np.vstack([seen_n, np.tile([0, -1, 0], (len(floor), 1))])
    rec = pcl.ObjRecRANSAC(pair_width=0.1, voxel_size=0.015)
    rec.add_Model(pcl.PointCloud(pts), _normals(nrm), "blob")
    found = rec.recognize(pcl.PointCloud(scene), _normals(scene_n))
    assert found, "nothing recognised"
    name, pose, confidence = max(found, key=lambda f: f[2])
    assert name == "blob" and confidence > 0.3
    assert np.allclose(pose[:3, 3], where, atol=0.01)
    # the blob is symmetric under a half turn about its x axis, so compare
    # where the model lands rather than the rotation
    placed = pts @ pose[:3, :3].T + pose[:3, 3]
    target = pts @ rot.T + where
    assert np.abs(np.sort(placed, 0) - np.sort(target, 0)).mean() < 0.005


def test_obj_rec_ransac_arguments():
    pts, nrm = _blob(20)
    rec = pcl.ObjRecRANSAC(0.1, 0.01)
    with pytest.raises(RuntimeError, match="add_Model"):
        rec.recognize(pcl.PointCloud(pts), _normals(nrm))
    with pytest.raises(ValueError, match="normals"):
        rec.add_Model(pcl.PointCloud(pts), _normals(nrm[:5]), "a")
    rec.add_Model(pcl.PointCloud(pts), _normals(nrm), "a")
    with pytest.raises(ValueError, match="already"):
        rec.add_Model(pcl.PointCloud(pts), _normals(nrm), "a")
    with pytest.raises(ValueError):
        pcl.ObjRecRANSAC(0.0, 0.01)


# --- implicit shape model -------------------------------------------------------------------


@pytest.fixture(scope="module")
def ism_scene():
    blob, blob_n = _blob(50)
    cyl, cyl_n = _cylinder()      # its normals include exact (1, 0, 0)
    blob_n = blob_n.copy()
    blob_n[5] = np.nan            # PCL would vote with it
    # one example per class: PCL's 184 clusters sometimes leave a class
    # without a word of its own (no votes at all); 400 never did in 8 runs
    ism = pcl.ImplicitShapeModelEstimation(feature_radius=0.05,
                                           sampling_size=0.02,
                                           number_of_clusters=400)
    ism.add_TrainingCloud(pcl.PointCloud(blob), _normals(blob_n), 0)
    ism.add_TrainingCloud(pcl.PointCloud(cyl), _normals(cyl_n), 1)
    ism.train()
    at_blob, at_cyl = np.array([1.0, 0.5, 2.0]), np.array([-1.0, 0.0, 2.0])
    scene = np.vstack([blob + at_blob, cyl + at_cyl]).astype(np.float32)
    normals = _normals(np.vstack([np.nan_to_num(blob_n), cyl_n]))
    return ism, pcl.PointCloud(scene), normals, at_blob, at_cyl


def test_ism_votes_for_each_class_at_its_object(ism_scene):
    ism, scene, normals, at_blob, at_cyl = ism_scene
    assert ism.number_of_classes == 2 and len(ism.sigmas) == 2
    # sigma is a tenth of each class's diameter
    assert ism.sigmas[0] == pytest.approx(0.2 * 0.13, rel=0.3)
    # k-means starts from random centres, so the peak moves a little
    # between trainings; it stays on its own object, 2 m from the other
    for class_id, centre in ((0, at_blob), (1, at_cyl)):
        peaks = ism.find_Objects(scene, normals, class_id)
        assert peaks.shape[1] == 4 and len(peaks) >= 1
        assert np.all(np.diff(peaks[:, 3]) <= 0)
        assert np.linalg.norm(peaks[0, :3] - centre) < 0.15


def test_ism_model_round_trip(ism_scene, tmp_path):
    ism, scene, normals, _, at_cyl = ism_scene
    path = tmp_path / "model.txt"
    ism.save_Model(path)
    other = pcl.ImplicitShapeModelEstimation(feature_radius=0.05,
                                             sampling_size=0.02,
                                             number_of_clusters=7)
    other.load_Model(path)
    assert other.number_of_classes == 2
    # the file stores text, so the weights come back rounded
    assert np.allclose(other.find_Objects(scene, normals, 1)[0],
                       ism.find_Objects(scene, normals, 1)[0], atol=1e-5)
    with pytest.raises(IOError):
        other.load_Model(tmp_path / "missing.txt")


def test_ism_arguments(ism_scene):
    ism, scene, normals, _, _ = ism_scene
    fresh = pcl.ImplicitShapeModelEstimation()
    with pytest.raises(RuntimeError, match="add_TrainingCloud"):
        fresh.train()
    with pytest.raises(RuntimeError, match="train"):
        fresh.find_Objects(scene, normals, 0)
    with pytest.raises(ValueError, match="class_id"):
        ism.find_Objects(scene, normals, 2)
    with pytest.raises(ValueError, match="finite"):
        fresh.add_TrainingCloud(
            pcl.PointCloud(np.ones((3, 3), np.float32)),
            _normals(np.full((3, 3), np.nan, np.float32)), 0)


# --- template matching ------------------------------------------------------------------------


W, H, F = 160, 120, 100.0


def _patch_scene(ox, oy):
    """A wall at z = 1.5 and a raised 40 x 40 patch (z = 1.2, top-left
    pixel (ox, oy)) carrying a red cross on green."""
    v, u = np.mgrid[0:H, 0:W]
    inside = (u >= ox) & (u < ox + 40) & (v >= oy) & (v < oy + 40)
    z = np.where(inside, 1.2, 1.5).astype(np.float32)
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z])
    du, dv = u - ox, v - oy
    cross = inside & (((du >= 16) & (du < 24)) | ((dv >= 16) & (dv < 24)))
    rgba = np.full((H, W, 4), 90, np.uint8)
    rgba[..., 3] = 255
    rgba[inside] = (20, 200, 30, 255)
    rgba[cross] = (230, 20, 30, 255)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz.astype(np.float32), rgba)
    return cloud, inside


def test_line_rgbd_finds_the_patch():
    frame, mask = _patch_scene(30, 30)
    matcher = pcl.LineRGBD()
    assert matcher.add_Template(frame, mask, object_id=7) == 0
    for ox, oy in ((30, 30), (90, 50), (10, 70)):
        found = matcher.detect(_patch_scene(ox, oy)[0])
        assert found, "no detection at (%d, %d)" % (ox, oy)
        best = max(found, key=lambda d: d["response"])
        assert best["object_id"] == 7 and best["template_id"] == 0
        x, y, w, h = best["region"]
        assert (w, h) == (40, 40)
        assert abs(x - ox) <= 8 and abs(y - oy) <= 8
        # the box comes from the template's own points: one patch wide
        assert 0.4 < best["bounding_box"][3] < 0.5
    assert matcher.detect_semi_scale_invariant(_patch_scene(90, 50)[0])
    with pytest.raises(ValueError, match="no pixel"):
        matcher.add_Template(frame, np.zeros((H, W), bool))
    with pytest.raises(ValueError, match="shape"):
        matcher.add_Template(frame, mask[:10])
    with pytest.raises(RuntimeError, match="add_Template"):
        pcl.LineRGBD().detect(frame)


def test_dotmod_finds_the_patch():
    frame, _ = _patch_scene(32, 32)
    dot = pcl.DOTMOD(40, 40, bin_size=8)
    assert dot.add_Template(frame, 32, 32) == 0
    for ox, oy in ((32, 32), (88, 48), (8, 72)):
        found = dot.detect(_patch_scene(ox, oy)[0], 0.8)
        assert found[0] == (ox, oy, 0, pytest.approx(1.0))
    with pytest.raises(ValueError, match="does not fit"):
        dot.add_Template(frame, W - 40, 0)
    with pytest.raises(ValueError):
        pcl.DOTMOD(4, 40, bin_size=8)
