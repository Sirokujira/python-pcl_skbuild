"""TextureMapping, AgastKeypoint2D's detector choice (and PCL's stale
detector), and the YUV 4:2:2 / Bayer .pclzf formats."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")


# --- texture mapping -------------------------------------------------------------


def _grid_mesh(n=10):
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    pts = np.c_[xx.ravel(), yy.ravel(), np.zeros(n * n)].astype(np.float32)
    tris = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            tris += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    return pts, np.array(tris)


def _camera(x, f=200.0, w=320, h=240):
    """Above the z = 0 square at height 2, looking down (-z world)."""
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = (x, 0.0, 2.0)
    return {"pose": pose, "focal_length": f, "width": w, "height": h}


def test_each_camera_takes_what_it_sees():
    pts, tris = _grid_mesh()
    cams = [_camera(-0.8), _camera(0.8)]
    pcl.set_verbosity_level(pcl.L_ERROR)    # PCL logs each camera at INFO
    try:
        groups = pcl.texture_mesh_with_cameras(pcl.PointCloud(pts), tris, cams)
    finally:
        pcl.set_verbosity_level(pcl.L_INFO)
    assert len(groups) == 3
    counts = [len(f) for f, _ in groups]
    assert sum(counts) == len(tris) and counts[0] > counts[1] > 0
    assert counts[2] == 0
    # every face is accounted for once
    seen = np.vstack([f for f, _ in groups if len(f)])
    assert sorted(map(tuple, seen)) == sorted(map(tuple, tris))
    # UVs follow the pinhole model: u = (f x / z + cx) / w,
    # v = 1 - (f y / z + cy) / h in the camera frame
    faces, uv = groups[0]
    pose = cams[0]["pose"]
    cam_pts = (pts - pose[:3, 3]) @ pose[:3, :3]
    expect_u = (200 * cam_pts[:, 0] / cam_pts[:, 2] + 160) / 320
    expect_v = 1 - (200 * cam_pts[:, 1] / cam_pts[:, 2] + 120) / 240
    assert np.allclose(uv[..., 0], expect_u[faces], atol=1e-5)
    assert np.allclose(uv[..., 1], expect_v[faces], atol=1e-5)


def test_unseen_faces_get_minus_one():
    pts, tris = _grid_mesh(4)
    away = _camera(0.0)
    away["pose"] = away["pose"].copy()
    away["pose"][:3, :3] = np.eye(3)          # looking up, away from it
    groups = pcl.texture_mesh_with_cameras(pcl.PointCloud(pts), tris, [away])
    assert len(groups[0][0]) == 0 and len(groups[1][0]) == len(tris)
    assert np.all(groups[1][1] == -1)
    with pytest.raises(ValueError):
        pcl.texture_mesh_with_cameras(pcl.PointCloud(pts), tris + 100, [away])
    with pytest.raises(ValueError):
        pcl.texture_mesh_with_cameras(pcl.PointCloud(pts), tris, [])


# --- AGAST -------------------------------------------------------------------------


def _frame(h, w, squares, value=255):
    img = np.zeros((h, w), np.uint8)
    for r, c in squares:
        img[r:r + 8, c:c + 8] = value
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xyz = np.dstack([u, v, np.ones((h, w))]).astype(np.float32)
    rgba = np.dstack([img, img, img, np.full_like(img, 255)])
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)
    return cloud


def test_agast_handles_frames_of_another_size():
    """PCL keeps the detector it built for the first frame; a wider
    second frame then had no corners at all."""
    small = _frame(40, 40, [(10, 10)])
    big = _frame(80, 120, [(30, 90)])
    agast = pcl.AgastKeypoint2D()
    agast.set_InputCloud(small)
    assert len(agast.compute()) == 4
    agast.set_InputCloud(big)
    corners = agast.compute()
    assert len(corners) == 4 and np.all(corners[:, 0] >= 89)


def test_agast_threshold_and_detectors():
    grey = _frame(60, 60, [(20, 20)], value=100)
    agast = pcl.AgastKeypoint2D(grey)
    assert len(agast.compute()) == 4
    agast.set_Threshold(150)            # above the square's contrast
    assert len(agast.compute()) == 0
    for detector in ("5_8", "oast_9_16"):
        assert len(pcl.AgastKeypoint2D(grey, detector=detector).compute()) == 4
    with pytest.raises(ValueError):
        pcl.AgastKeypoint2D(detector="fast")


# --- YUV 4:2:2 and Bayer .pclzf ---------------------------------------------------------


def test_pclzf_colour_formats(tmp_path):
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0] = 200
    pcl.write_pclzf_rgb(tmp_path / "a.pclzf", rgb)
    assert np.array_equal(pcl.read_pclzf_image(tmp_path / "a.pclzf"), rgb)

    uyvy = np.zeros((6, 8, 2), np.uint8)
    uyvy[..., 0] = 128                  # no chroma: grey
    uyvy[..., 1] = 100
    pcl.write_pclzf_yuv422(tmp_path / "b.pclzf", uyvy)
    assert np.all(pcl.read_pclzf_image(tmp_path / "b.pclzf") == 100)
    uyvy[:, 1::2, 0] = 200              # V up: towards red
    pcl.write_pclzf_yuv422(tmp_path / "c.pclzf", uyvy)
    red = pcl.read_pclzf_image(tmp_path / "c.pclzf").astype(int)
    assert np.all(red[..., 0] > red[..., 1])

    pcl.write_pclzf_bayer8(tmp_path / "d.pclzf", np.full((8, 8), 90, np.uint8))
    assert np.all(pcl.read_pclzf_image(tmp_path / "d.pclzf") == 90)

    pcl.write_pclzf_depth(tmp_path / "e.pclzf", np.full((6, 8), 1000, np.uint16))
    with pytest.raises(ValueError, match="depth"):
        pcl.read_pclzf_image(tmp_path / "e.pclzf")
    with pytest.raises(IOError):
        pcl.read_pclzf_image(tmp_path / "missing.pclzf")
    with pytest.raises(ValueError):
        pcl.write_pclzf_yuv422(tmp_path / "f.pclzf", np.zeros((4, 5, 2), np.uint8))
